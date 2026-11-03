from collections.abc import Callable
from typing import Literal

import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData


def leiden_clustering(adata: AnnData, flavor: Literal["leidenalg", "igraph"] = "igraph", **kwargs):
    adata = adata.copy()
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, flavor=flavor, **kwargs)
    return adata.obs["leiden"].values


def kmeans_clustering(adata: AnnData, **kwargs):
    from sklearn.cluster import KMeans

    kmeans = KMeans(**kwargs)
    return kmeans.fit_predict(adata.X)


METHODS_DICT = {
    "leiden": leiden_clustering,
    "kmeans": kmeans_clustering,
}


def cluster_embeddings(
    sdata: SpatialData | None,
    element: AnnData | str,
    method: Callable | str = "leiden",
    key_added: str = "cluster",
    **method_kwargs: str,
) -> None:
    """Create clusters of the patches embeddings (obtained from [sopa.compute_embeddings][]).

    Info:
        The clusters are added to the `key_added` column of the "inference_patches" shapes (`key_added='cluster'` by default).

    Args:
        sdata: A `SpatialData` object. Can be `None` if element is an `AnnData` object.
        element: The `AnnData` containing the embeddings, or the name of the element
        method: Callable that takes as an AnnData object and returns an array of clusters of size `n_obs`, or an available method name (`leiden` or `kmeans`)
        key_added: The key containing the clusters to be added to the `element.obs`
        method_kwargs: kwargs provided to the method callable
    """
    if isinstance(element, str):
        element: AnnData = sdata.tables[element]

    if isinstance(method, str):
        assert method in METHODS_DICT, f"Method {method} is not available. Use one of: {', '.join(METHODS_DICT.keys())}"
        method = METHODS_DICT[method]

    element.obs[key_added] = method(element, **method_kwargs)
    element.obs[key_added] = element.obs[key_added].astype("category")
