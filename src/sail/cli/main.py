import typer
import torch, tqdm, os
import pandas as pd
from ..core.registries import ModelRegistry, DatasetRegistry
from ..training.trainer import Trainer, TrainConfig
from ..explain import Simba
from ..data.adapters import SimbaJSONDataset, JSONGeoAdapter, resolve_json_paths
from ..utils.utils import *
from ..data.instance_index import *
from ..explain.utils import *



app = typer.Typer(help="SIMBA CLI")

@app.command()
def train(
    model: str = "tang2015",
    dataset: str = "json",   # 'json' triggers generic loader
    data_root: str = typer.Option("", help="Folder containing JSONs and images"),
    prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl' or 'western_africa'"),
    with_neighbors: bool = False,
    batch_size: int = 16,
    epochs: int = 2,
    lr: float = 1e-3,
    ckpt_dir: str = "artifacts/checkpoints",
):

    saved_batch_size = batch_size
    if model == "geoconv":
        batch_size = 1

    
    print("WITH NEIGHBORS: ", with_neighbors)
    # Build dataset
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = JSONGeoAdapter(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            batch_size=batch_size,
            ckpt_dir = ckpt_dir
        )
    else:
        ds = DatasetRegistry.get(dataset)()

    # Build model
    mw = ModelRegistry.get(model)()

    # Fail early if Tang model requires neighbors but user disabled it
    if model == "tang2015" and not with_neighbors:
        raise typer.BadParameter("Tang2015 model requires neighbors. Use --with-neighbors true.")

    cfg = TrainConfig(epochs=epochs, lr=lr, ckpt_dir=ckpt_dir)
    Trainer(mw, ds, cfg, model, saved_batch_size).fit()



@app.command()
def validate(model: str = "tang2015",
             dataset: str = "json",   # 'json' triggers generic loader
             data_root: str = typer.Option("", help="Folder containing JSONs and images"),
             prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl' or 'western_africa'"),
             with_neighbors: bool = True,
             ckpt_dir: str = "artifacts/checkpoints",
             device = "cuda",
             max_epoch = None):

    with_neighbors = False

    # Load trained model
    mw = ModelRegistry.get(model)()#.build()
    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)
    mw.load(path)

    # Build dataset    
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = SimbaJSONDataset(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            ckpt_dir = ckpt_dir,
            validate = True,
            
        )
    else:
        ds = DatasetRegistry.get(dataset)()


    mw.net = mw.net.to(device).eval()
    

    imnames, preds, labels = [], [], []
    for c, batch in tqdm.tqdm(enumerate(ds), desc = "Validating"):

        batch = {k: (v.to(device).unsqueeze(0) if hasattr(v, "to") else v) for k,v in batch.items()}

        out = mw.forward(batch)
        
        # print(, batch["label"])

        imnames.append(batch["image_name"])
        preds.append(out.item())
        labels.append(batch["label"].item())

        # print(imnames, preds, labels)
        
        
        if c % 10:
            df = pd.DataFrame()
            df["name"], df["pred"], df["label"] = imnames, preds, labels
            df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))

        df = pd.DataFrame()
        df["name"], df["pred"], df["label"] = imnames, preds, labels
        df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))  

        # print(c, len(ds), end = "\r")
        
        
        # print(i)
        # break


# @app.command()
# def explain(model_ckpt: str = "artifacts/checkpoints/model.pt",
#            dataset: str = "toygeo",
#            delta_small: int = 250, delta_large: int = 1000,
#            n_batches: int = 5):
#     # load model & data
#     mw = ModelRegistry.get("from_ckpt")()
#     mw.load(model_ckpt)
#     ds = DatasetRegistry.get(dataset)()
#     expl = SimbaExplainer(mw, ds)
#     res = expl.global_sensitivity(deltas_m=(delta_small, delta_large), n_batches=n_batches)
#     typer.echo(res)



@app.command("explain")
def explain(model: str = "tang2015",
             dataset: str = "json",   # 'json' triggers generic loader
             data_root: str = typer.Option("", help="Folder containing JSONs and images"),
             prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl' or 'western_africa'"),
             # with_neighbors: bool = True,
             ckpt_dir: str = "artifacts/checkpoints",
                with_neighbors: bool = typer.Option(
                    True, "--with-neighbors/--no-neighbors",
                    help="Include nearest-neighbor context in the explanation.",
                    show_default=True,
                ),
                distances: str = "0.25,1,5",
                index: int = 0,
                out_csv: str = "artifacts/sensitivity_instance.csv",
            device = "cuda",
            max_instances: int = None,
           ):


    print("IN EXPLAIN")

    # Load trained model
    mw = ModelRegistry.get(model)()#.build()
    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)
    mw.load(path)

    df = None

    print("1")

    # Build dataset    
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = SimbaJSONDataset(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            ckpt_dir = ckpt_dir,
            validate = False,
            
        )
    else:
        ds = DatasetRegistry.get(dataset)()


    print("2")


    mw.net = mw.net.to(device).eval()

    distances_km = [float(x) for x in distances.split(",")]


        # Run analyzer
    analyzer = Simba(model = mw.net,
                     baseline_image = None,
                     baseline_coords = None,
                     label = None,
                     additional_model_inputs = None,# COME BACK AND MAKE THIS MORE FELXIBLE TO THE MODEL LATER ON!!!!
                     ckpt_dir = ckpt_dir
                    )  # uses GPU if available

    analyzer.explain_global_from_list(ds,
                                      distances_km = distances_km,
                                      max_instances = max_instances)

    

    # distances_km = [float(x) for x in distances.split(",")]    

    # imnames, preds, labels = [], [], []
    # for c, batch in tqdm.tqdm(enumerate(ds), desc = "Explaining"):

    #     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in batch.items()}

    #     if not with_neighbors:
    #         ami = None
    #     else:
    #         ami = [batch["neighbor_images"], batch["neighbor_mask"]]
    
    #     # Run analyzer
    #     analyzer = Simba(model = mw.net,
    #                      baseline_image = batch["image"],
    #                      baseline_coords = batch["coords"].cpu().numpy(),
    #                      label = batch["label"],
    #                      additional_model_inputs = ami # COME BACK AND MAKE THIS MORE FELXIBLE TO THE MODEL LATER ON!!!!
    #                     )  # uses GPU if available


    #     if c == 0:
    #         df = analyzer.run_instance_analysis(
    #             # baseline_image=img,
    #             # baseline_coords=crd,
    #             distances_km=distances_km,
    #             # neighbor_images=nb,
    #             # neighbor_mask=nm,
    #             # regression=True,  # your current models are regression
    #         )
    #         df["image_name"] = batch["image_name"]
    #     else:
    #         df_new = analyzer.run_instance_analysis(
    #             # baseline_image=img,
    #             # baseline_coords=crd,
    #             distances_km=distances_km,
    #             # neighbor_images=nb,
    #             # neighbor_mask=nm,
    #             # regression=True,  # your current models are regression
    #         )
    #         df_new["image_name"] = batch["image_name"]
    #         # df.apend(df_new)
    #         df = pd.concat([df, df_new], axis=0)



    #     # if c > 5:

        
    #         # print(batch.keys())
            
    #     analyzer.export_results(df, out_csv)
    #     typer.echo(f"Saved instance sensitivity to {out_csv}")
        # break


        

        # batch = {k: (v.to(device).unsqueeze(0) if hasattr(v, "to") else v) for k,v in batch.items()}

        # out = mw.forward(batch)
        
        # # print(, batch["label"])

        # imnames.append(batch["image_name"])
        # preds.append(out.item())
        # labels.append(batch["label"].item())

        # # print(imnames, preds, labels)
        
        
        # if c % 10:
        #     df = pd.DataFrame()
        #     df["name"], df["pred"], df["label"] = imnames, preds, labels
        #     df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))

        # df = pd.DataFrame()
        # df["name"], df["pred"], df["label"] = imnames, preds, labels
        # df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))  

        # # print(c, len(ds), end = "\r")
        
        
        # # print(i)
        # # break


# @app.command("explain-instance")  # hyphenated CLI name
# def explain_instance(
#     model: str = "tang2015",
#     # model_ckpt: str = typer.Option(..., "--model-ckpt", help="Path to trained checkpoint"),
#     ckpt_dir: str = "artifacts/checkpoints",
#     dataset: str = "json",
#     data_root: str = typer.Option("", help="Folder with JSONs & images"),
#     prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl'"),
#     with_neighbors: bool = typer.Option(
#         True, "--with-neighbors/--no-neighbors",
#         help="Include nearest-neighbor context in the explanation.",
#         show_default=True,
#     ),
#     distances: str = "0.25,1,5",
#     index: int = 0,
#     out_csv: str = "artifacts/sensitivity_instance.csv",
#     device = "cuda"
# ):

#     # Build dataset    
#     if dataset == "json":
#         if not data_root or not prefix:
#             raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
#         ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
#         ds = SimbaJSONDataset(
#             root_dir=data_root,
#             ys_path=ys,
#             coords_path=coords,
#             dup_path=dup,
#             ckpt_dir = ckpt_dir,
#             validate = False
#         )
#     else:
#         ds = DatasetRegistry.get(dataset)()    

#     index = InstanceIndex(coords)

#     print(index.coords_deg)

#     lat, lon = 7.22417677648748, -10.721340976497585

#     distances_km = [float(x) for x in distances.split(",")]

#     nearby = index.nearest(lat, lon)

#     print("Nearby: ", nearby)

#     # Load trained model
#     mw = ModelRegistry.get(model)()#.build()
#     epoch, path = highest_epoch(ckpt_dir)
#     print(epoch, path)
#     mw.load(path)

#     print("Index: ", ds.items.index(nearby[0][0]))

#     index = ds.items.index(nearby[0][0])

#     batch = ds[index]

#     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in batch.items()}

#     print("WN: ", with_neighbors)

#     if not with_neighbors:
#         ami = None
#     else:
#         ami = [batch["neighbor_images"], batch["neighbor_mask"]]

#     # Run analyzer
#     analyzer = Simba(model = mw.net,
#                      baseline_image = batch["image"],
#                      baseline_coords = batch["coords"].cpu().numpy(),
#                      label = batch["label"],
#                      additional_model_inputs = ami # COME BACK AND MAKE THIS MORE FELXIBLE TO THE MODEL LATER ON!!!!
#                     )  # uses GPU if available

#     print("distances_km: ", distances_km)
    
#     df = analyzer.run_instance_analysis(
#         # baseline_image=img,
#         # baseline_coords=crd,
#         distances_km=distances_km,
#         # neighbor_images=nb,
#         # neighbor_mask=nm,
#         # regression=True,  # your current models are regression
#     )
#     analyzer.export_results(df, out_csv)
#     typer.echo(f"Saved instance sensitivity to {out_csv}")





@app.command("explain-instance")  # hyphenated CLI name
def explain_instance(
    model: str = "tang2015",
    ckpt_dir: str = "artifacts/checkpoints",
    dataset: str = "json",
    data_root: str = typer.Option("", help="Folder with JSONs & images"),
    prefix: str = typer.Option("", help="Dataset prefix, e.g., 'phl'"),
    with_neighbors: bool = typer.Option(
        True, "--with-neighbors/--no-neighbors",
        help="Include nearest-neighbor context in the explanation.",
        show_default=True,
    ),
    input_coords: str = "14.60827772362026,121.00946800454903",
    distances: str = "0.25,1,5",
    device = "cuda"
):

    with_neighbors = False

    # Build dataset    
    if dataset == "json":
        if not data_root or not prefix:
            raise typer.BadParameter("When dataset='json', you must pass --data-root and --prefix.")
        ys, coords, dup = resolve_json_paths(data_root, prefix, with_neighbors=with_neighbors)
        ds = SimbaJSONDataset(
            root_dir=data_root,
            ys_path=ys,
            coords_path=coords,
            dup_path=dup,
            ckpt_dir = ckpt_dir,
            validate = False,
        )
    else:
        ds = DatasetRegistry.get(dataset)()    

    # Create ball-tree for looking up input coordinates
    index = InstanceIndex(coords)

    

    # reformat input coordinates
    input_coords = input_coords.split(",")
    lat, lon = float(input_coords[0]), float(input_coords[1])

    # Get dataset instances nearby
    distances_km = [float(x) for x in distances.split(",")]
    nearby = index.nearest(lat, lon)

    # Load trained model
    mw = ModelRegistry.get(model)()#.build()
    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)
    mw.load(path)

    # Load batch
    index = ds.items.index(nearby[0][0])
    batch = ds[index]
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k,v in batch.items()}

    if not with_neighbors:
        ami = None
    else:
        ami = [batch["neighbor_images"], batch["neighbor_mask"]]

    # Run analyzer
    analyzer = Simba(model = mw.net,
                     baseline_image = batch["image"],
                     baseline_coords = batch["coords"].cpu().numpy(),
                     label = batch["label"],
                     additional_model_inputs = ami, # COME BACK AND MAKE THIS MORE FELXIBLE TO THE MODEL LATER ON!!!!
                     ckpt_dir = ckpt_dir
                    )  # uses GPU if available

    print("distances_km: ", distances_km)

    analyzer.explain_instance(distances_km = distances_km)
    
    # df = analyzer.run_instance_analysis(distances_km = distances_km)
    # analyzer.export_results(df)
    
    # # Get, save and print stats    
    # metrics = analyzer.summarize_variogram_from_df(df)
    # analyzer.plot_variogram_summary()
    # res = analyzer.simba_variogram_with_null(distances_km)
    # analyzer.print_null_variogram_summary(res)
    # analyzer.export_results(df)
    # typer.echo(f"Saved instance sensitivity metrics!")



if __name__ == "__main__":
    app()
