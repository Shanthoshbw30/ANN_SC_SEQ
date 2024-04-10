{\rtf1\ansi\ansicpg1252\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import anndata as ad\
import matplotlib.pyplot as plt\
import scanpy as sc\
import plotly.graph_objs as go\
import pandas as pd\
import numpy as np\
import marsilea as ma\
import marsilea.plotter as mp\
\
def load_and_preprocess_data():\
    sc.settings.verbosity = 3\
    sc.logging.print_header()\
    sc.settings.set_figure_params(dpi=200, facecolor="white")\
    \
    results_file = "write/pbmc3k.h5ad"\
    \
    adata = sc.datasets.pbmc3k_processed().raw.to_adata()\
    adata.obs_names_make_unique()\
    adata.var_names_make_unique()\
    \
    return adata, results_file\
\
def filter_and_qc(adata):\
    sc.pp.filter_cells(adata, min_genes=200)\
    sc.pp.filter_genes(adata, min_cells=3)\
    \
    adata.var["mt"] = adata.var_names.str.startswith("MT-")\
    \
    sc.pp.calculate_qc_metrics(\
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True\
    )\
    \
    return adata\
\
def preprocess_data(adata):\
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]\
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()\
    adata.layers["counts"] = adata.X.copy()\
    \
    sc.pp.normalize_total(adata, target_sum=1e4)\
    sc.pp.log1p(adata)\
    \
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)\
    adata = adata[:, adata.var.highly_variable]\
    \
    sc.pp.scale(adata, max_value=10)\
    sc.tl.pca(adata, svd_solver="arpack")\
    \
    return adata\
\
def run_clustering_and_embedding(adata):\
    sc.pp.neighbors(adata)\
    sc.tl.umap(adata)\
    \
    sc.tl.leiden(\
        adata,\
        resolution=0.9,\
        random_state=0,\
        flavor="igraph",\
        n_iterations=2,\
        directed=False\
    )\
    \
    sc.tl.louvain(\
        adata,\
        resolution=0.9,\
        random_state=0,\
        flavor="igraph",\
        directed=False,\
    )\
    \
    sc.tl.paga(adata)\
    \
    return adata\
\
def run_differential_expression_analysis(adata):\
    sc.tl.rank_genes_groups(adata, "leiden", method="t-test")\
    sc.tl.rank_genes_groups(adata, groupby='leiden', method='logreg', solver='saga')\
    \
    return adata\
\
def extract_de_results(adata, group_number='0'):\
    if 'rank_genes_groups' not in adata.uns:\
        print("Differential expression analysis has not been run.")\
        return None\
    \
    if not isinstance(group_number, str):\
        group_number = str(group_number)  # Ensure group_number is a string\
    \
    names_data = adata.uns['rank_genes_groups']['names']\
    \
    if isinstance(names_data, np.recarray):\
        # Check if 'names' is a recarray\
        names_dict = \{str(group): names_data[group] for group in names_data.dtype.names\}\
        if group_number not in names_dict:\
            print(f"No results found for group number \{group_number\}.")\
            return None\
    else:\
        print("Unexpected data format for 'names'.")\
        return None\
    \
    if 'logfoldchanges' not in adata.uns['rank_genes_groups']:\
        print(f"No log-fold changes calculated for group number \{group_number\}.")\
        return None\
    \
    de_results = pd.DataFrame(\{\
        'gene_names': names_dict[group_number],\
        'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges'][group_number],\
        'pvals': adata.uns['rank_genes_groups']['pvals'][group_number],\
        'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][group_number]\
    \})\
    \
    de_results['pvals_adj'] = de_results['pvals_adj'].replace(0, np.finfo(float).eps)\
    de_results['-log10pvals_adj'] = -np.log10(de_results['pvals_adj'])\
    max_log_value = -np.log10(np.finfo(float).eps)\
    de_results['-log10pvals_adj'] = de_results['-log10pvals_adj'].replace(np.inf, max_log_value)\
    de_results['log2foldchanges'] = np.log2(de_results['logfoldchanges'])\
    \
    return de_results\
\
def create_i_volcano_plot(adata, group, pval_cutoff=0.05, log2fc_cutoff=1, top_n=3):\
    de_results = extract_de_results(adata, group)\
    \
    # Get top significant genes\
    sig_genes = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'].abs() >= log2fc_cutoff)]\
    sig_upregulated = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'] > log2fc_cutoff)]\
    sig_downregulated = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'] < -log2fc_cutoff)]\
    top_up_genes = sig_upregulated.nlargest(top_n, '-log10pvals_adj')\
    top_down_genes = sig_downregulated.nlargest(top_n, '-log10pvals_adj')\
    \
    # Create the interactive volcano plot\
    fig = px.scatter(\
        de_results, \
        x='logfoldchanges', \
        y=-np.log10(de_results['pvals_adj']),\
        color=np.where((de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'].abs() >= log2fc_cutoff), 'DE', 'not DE'),\
        hover_data=['gene_names']\
    )\
    \
    # Add horizontal line for p-value cutoff\
    fig.add_shape(\
        type='line',\
        line=dict(dash='dash', color='black', width=1),\
        x0=-max(de_results['logfoldchanges']), x1=max(de_results['logfoldchanges']),\
        y0=-np.log10(pval_cutoff), y1=-np.log10(pval_cutoff)\
    )\
\
    # Add vertical lines for log2fc cutoffs\
    fig.add_shape(\
        type='line',\
        line=dict(dash='dash', color='black', width=1),\
        x0=-log2fc_cutoff, x1=-log2fc_cutoff,\
        y0=0, y1=max(de_results['-log10pvals_adj'])\
    )\
    fig.add_shape(\
        type='line',\
        line=dict(dash='dash', color='black', width=1),\
        x0=log2fc_cutoff, x1=log2fc_cutoff,\
        y0=0, y1=max(de_results['-log10pvals_adj'])\
    )\
\
    # Add upregulated genes\
    fig.add_trace(go.Scatter(\
        x=sig_upregulated['logfoldchanges'],\
        y=-np.log10(sig_upregulated['pvals_adj']),\
        mode='markers',\
        marker=dict(color='green', size=7, symbol='triangle-up'),\
        name='Upregulated',\
        text=sig_upregulated['gene_names'],\
        hoverinfo='text'\
    ))\
\
    # Add downregulated genes\
    fig.add_trace(go.Scatter(\
        x=sig_downregulated['logfoldchanges'],\
        y=-np.log10(sig_downregulated['pvals_adj']),\
        mode='markers',\
        marker=dict(color='orange',symbol='triangle-down', size=7),\
        name='Downregulated',\
        text=sig_downregulated['gene_names'],\
        hoverinfo='text'\
    ))\
\
    fig.add_trace(go.Scatter(\
        x=sig_genes['log2foldchanges'],\
        y=-np.log10(sig_genes['pvals_adj']),\
        mode='markers',\
        name='not DE',\
        marker=dict(color='grey', size=5),\
        text=de_results['gene_names'],\
        hoverinfo='text'\
    ))\
    \
    # Set layout properties\
    fig.update_layout(\
        title=f'Interactive Volcano Plot (Group \{group\})',\
        xaxis_title='log2 Fold Change',\
        yaxis_title='-log10 Adjusted p-value',\
        template='plotly_white', \
        width=1000,  # Set the width of the plot\
        height=600,  # Set the height of the plot\
        xaxis=dict(range=[-7, 7]),\
        yaxis=dict(range=[0, 15])\
    )\
\
    # Add annotations for top significant genes\
    for i, row in top_up_genes.iterrows():\
        fig.add_annotation(\
            x=row['logfoldchanges'],\
            y=row['-log10pvals_adj'],\
            text=row['gene_names'],\
            showarrow=True,\
            arrowhead=1,\
            arrowsize=2,\
            arrowwidth=1,\
            arrowcolor='black'\
        )   \
        \
    for i, row in top_down_genes.iterrows():\
        fig.add_annotation(\
            x=row['logfoldchanges'],\
            y=row['-log10pvals_adj'],\
            text=row['gene_names'],\
            showarrow=True,\
            arrowhead=1,\
            arrowsize=2,\
            arrowwidth=1,\
            arrowcolor='black'\
        )\
    \
    # Show figure\
    fig.show()\
\
def main():\
    adata, results_file = load_and_preprocess_data()\
    adata = filter_and_qc(adata)\
    adata = preprocess_data(adata)\
    adata = run_clustering_and_embedding(adata)\
    adata = run_differential_expression_analysis(adata)\
    groups = adata.obs['leiden'].unique().tolist()\
    for group in groups:\
        create_i_volcano_plot(adata, group)\
\
if __name__ == "__main__":\
    main()\
}