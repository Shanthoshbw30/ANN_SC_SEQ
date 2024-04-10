{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import marsilea as ma\
import marsilea.plotter as mp\
import scanpy as sc\
import numpy as np\
\
def create_scRNA_seq_plots(adata, cell_markers, uni_cells, cmapper):\
    exp = adata[:, markers].X.toarray()\
\
    m = ma.Heatmap(exp, cmap="viridis", height=3.5, width=3)\
    m.render()\
\
    m.hsplit(labels=adata.obs["louvain"], order=uni_cells)\
    m.render()\
\
    # Create plotters\
    chunk = mp.Chunk(uni_cells, rotation=0, align="center")\
    colors = mp.Colors(list(adata.obs["louvain"]), palette=cmapper)\
    label_markers = mp.Labels(markers)\
\
    # Add to the heatmap\
    m.add_left(colors, size=0.1, pad=0.1)\
    m.add_left(chunk)\
    m.add_top(label_markers, pad=0.1)\
    m.render()\
\
    m.add_dendrogram("right", add_base=False)\
    m.render()\
\
    m.add_legends()\
    m.add_title("Expression Profile")\
    m.render()\
\
    agg = sc.get.aggregate(adata[:, markers], by="louvain", func=["mean", "count_nonzero"])\
    agg.obs["cell_counts"] = adata.obs["louvain"].value_counts()\
\
    exp = agg.layers["mean"]\
    count = agg.layers["count_nonzero"]\
    cell_counts = agg.obs["cell_counts"].to_numpy()\
\
    h, w = exp.shape\
\
    m = ma.Heatmap(\
        exp,\
        height=h / 3,\
        width=w / 3,\
        cmap="Blues",\
        linewidth=0.5,\
        linecolor="lightgray",\
        label="Expression",\
    )\
    m.add_right(mp.Labels(agg.obs["louvain"], align="center"), pad=0.1)\
    m.add_top(mp.Labels(markers), pad=0.1)\
    m.vsplit(labels=cells, order=uni_cells)\
    m.add_top(mp.Chunk(uni_cells, fill_colors=cell_colors, rotation=90))\
    m.add_left(mp.Numbers(cell_counts, color="#EEB76B", label="Count"))\
    m.add_dendrogram("right", pad=0.1)\
    m.add_legends()\
    m.render()\
\
    size = count / cell_counts[:, np.newaxis]\
    m = ma.SizedHeatmap(\
        size=size,\
        color=exp,\
        cluster_data=size,\
        height=h / 3,\
        width=w / 3,\
        edgecolor="lightgray",\
        cmap="Blues",\
        size_legend_kws=dict(\
            colors="#538bbf",\
            title="Fraction of cells\\nin groups (%)",\
            labels=["20%", "40%", "60%", "80%", "100%"],\
            show_at=[0.2, 0.4, 0.6, 0.8, 1.0],\
        ),\
        color_legend_kws=dict(title="Mean expression\\nin group"),\
    )\
\
    m.add_top(mp.Labels(markers), pad=0.1)\
    m.add_top(mp.Chunk(uni_cells, fill_colors=cell_colors, rotation=90))\
    m.vsplit(labels=cells, order=uni_cells)\
\
    m.add_right(mp.Labels(agg.obs["louvain"], align="center"), pad=0.1)\
    m.add_left(mp.Numbers(cell_counts, color="#EEB76B", label="Count"), size=0.5, pad=0.1)\
    m.add_dendrogram("right", pad=0.1)\
    m.add_legends()\
    m.render()\
\
# Example usage:\
adata = sc.datasets.pbmc3k_processed().raw.to_adata()\
\
cell_markers = \{\
    "CD4 T cells": ["IL7R"],\
    "CD14+ Monocytes": ["CD14", "LYZ"],\
    "B cells": ["MS4A1"],\
    "CD8 T cells": ["CD8A"],\
    "NK cells": ["GNLY", "NKG7"],\
    "FCGR3A+ Monocytes": ["FCGR3A", "MS4A7"],\
    "Dendritic cells": ["FCER1A", "CST3"],\
    "Megakaryocytes": ["PPBP"],\
\}\
\
uni_cells = list(cell_markers.keys())\
cell_colors = [\
    "#568564",\
    "#FFF3A7",\
    "#F72464",\
    "#005585",\
    "#9876DE",\
    "#405559",\
    "#58DADA",\
    "#F85959",\
]\
cmapper = dict(zip(uni_cells, cell_colors))\
\
create_scRNA_seq_plots(adata, cell_markers, uni_cells, cmapper)\
}