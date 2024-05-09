import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np
import marsilea as ma
import marsilea.plotter as mp
import plotly.express as px
import plotly.graph_objs as go


def load_and_preprocess_data():
    
    """
    Load and preprocess the single-cell RNA-seq data.
    """
    
    #Predefining verbosity
    sc.settings.verbosity = 3
    sc.logging.print_header()
    sc.settings.set_figure_params(dpi=200, facecolor="white")
    
    #a dataset from 10x containing 68k cells from PBMC
    results_file = "write/pbmc3k.h5ad"
    
    adata = sc.datasets.pbmc3k_processed().raw.to_adata()
    
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    
    return adata, results_file




def filter_and_qc(adata):
    """
    Filter cells and genes based on quality control metrics.
    """    
    
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    
    return adata



def preprocess_data(adata):
    """
    Preprocess data by filtering, normalization, and gene selection.
    """    
    
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :].copy()
   

    # Saving count data
    adata.layers["counts"] = adata.X.copy()
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    
    return adata

def run_clustering_and_embedding(adata):
    
    """
    Run Clsutering and embedding.
    """
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    sc.tl.leiden(
        adata,
        resolution=0.9,
        random_state=0,
        flavor="igraph",
        n_iterations=2,
        directed=False
    )
    
    sc.tl.louvain(
        adata,
        resolution=0.9,
        random_state=0,
        flavor="igraph",
        directed=False,
    )
    
    sc.tl.paga(adata)
    
    return adata



def run_differential_expression_analysis(adata):
    
    """
    Run differential expression analysis.
    """    
    sc.tl.rank_genes_groups(adata, "leiden", method="t-test")
    sc.tl.rank_genes_groups(adata, "louvain", method="t-test")

    sc.tl.rank_genes_groups(adata, "leiden", method="logreg")
    sc.tl.rank_genes_groups(adata, "louvain", method="logreg")
    sc.tl.rank_genes_groups(adata, groupby='leiden', method='wilcoxon', groups=["0","1","2","3","4","5","6"], reference="1")
    sc.tl.rank_genes_groups(adata, "louvain", method="wilcoxon",groups=["0","1","2","3","4","5","6"], reference="1")
    adata.uns['rank_genes_groups']
    
    return adata



def extract_de_results(adata, groups):
    
    """
    Extract differential expression results.
    """
    
    if 'rank_genes_groups' in adata.uns:
        groups = adata.uns['rank_genes_groups']['names'].dtype.names
        print("Differential expression analysis has been run.")
        print("Available groups for differential expression:", groups)
    else:
        print("Differential expression analysis failed or results are not stored correctly.")
        return None

    all_de_results = []
    for group_number in groups:
        de_results = pd.DataFrame({
            'gene_names': adata.uns['rank_genes_groups']['names'][group_number],
            'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges'][group_number],
            'pvals': adata.uns['rank_genes_groups']['pvals'][group_number],
            'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][group_number]
        })
        
        # Replace zeros with a very small number before taking log10
        de_results['pvals_adj'] = de_results['pvals_adj'].replace(0, np.finfo(float).eps)
        
        # Calculate the -log10 of adjusted p-values
        de_results['-log10pvals_adj'] = -np.log10(de_results['pvals_adj'])
        
        # Replace `inf` values that might have resulted from very small p-values
        max_log_value = -np.log10(np.finfo(float).eps)
        de_results['-log10pvals_adj'] = de_results['-log10pvals_adj'].replace(np.inf, max_log_value)
        
        # Calculate the log2 fold changes
        de_results['log2foldchanges'] = np.log2(de_results['logfoldchanges'])
        
        all_de_results.append(de_results)

    # Concatenate all DataFrames in the list
    all_de_results = pd.concat(all_de_results, ignore_index=True)

    return all_de_results

    
    


def create_i_volcano_plot(adata, groups, pval_cutoff=0.05, log2fc_cutoff=1, top_n=10):
    """
    Create a dynamic volcano plot of Upregularted and Downlregulated Genes.
    """    
    
    # Extract differential expression results for the specified group
    de_results = extract_de_results(adata, groups)
    
    # Get top significant genes
    sig_genes = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'].abs() >= log2fc_cutoff)]
    sig_upregulated = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'] >= log2fc_cutoff)]
    sig_downregulated = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'] < -log2fc_cutoff)]
    top_up_genes = sig_upregulated.nlargest(top_n, '-log10pvals_adj')
    top_down_genes = sig_downregulated.nlargest(top_n, '-log10pvals_adj')
   

    # Create the interactive volcano plot
    fig = px.scatter(
        de_results, 
        x='logfoldchanges', 
        y=-np.log10(de_results['pvals_adj']),
        color=np.where((de_results['pvals_adj'] < pval_cutoff) & (de_results['logfoldchanges'].abs() >= log2fc_cutoff), 'DE', 'not DE'),
        hover_data=['gene_names']
    ) 
    # Add horizontal line for p-value cutoff
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=-max(de_results['logfoldchanges']), x1=max(de_results['logfoldchanges']),
        y0=-np.log10(pval_cutoff), y1=-np.log10(pval_cutoff)
    )
    # Add vertical lines for log2fc cutoffs
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=-log2fc_cutoff, x1=-log2fc_cutoff,
        y0=0, y1=max(de_results['-log10pvals_adj'])
    )
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=log2fc_cutoff, x1=log2fc_cutoff,
        y0=0, y1=max(de_results['-log10pvals_adj'])
    )
    # Add upregulated genes
    fig.add_trace(go.Scatter(
        x=sig_upregulated['logfoldchanges'],
        y=-np.log10(sig_upregulated['pvals_adj']),
        mode='markers',
        marker=dict(color='green', size=16, symbol='triangle-up'),
        name='Upregulated',
        text=sig_upregulated['gene_names'],
        hoverinfo='text'
    ))
    # Add downregulated genes
    fig.add_trace(go.Scatter(
        x=sig_downregulated['logfoldchanges'],
        y=-np.log10(sig_downregulated['pvals_adj']),
        mode='markers',
        marker=dict(color='orange',symbol='triangle-down', size=16),
        name='Downregulated',
        text=sig_downregulated['gene_names'],
        hoverinfo='text'
    )) 
    # Set layout properties
    fig.update_layout(
        title=f'Interactive Volcano Plot ',
        xaxis_title='log Fold Change',
        yaxis_title='-log10 Adjusted p-value',
        template='plotly_white', width=1000,  # Set the width of the plot
        height=600,  # Set the height of the plot
        xaxis=dict(range=[-7, 7]),
        yaxis=dict(range=[0, 15])   )

    # Show figure
    fig.show()




def create_interactive_volcano_plot(adata, groups, pval_cutoff=0.05, log2fc_cutoff=1, top_n=10):

    """
    Create a dynamic volcano plot of Differential expressed genes.
    """    
    
    
    # Get top significant genes
    de_results = extract_de_results(adata, groups)
    sig_genes = de_results[(de_results['pvals_adj'] < pval_cutoff) & (de_results['log2foldchanges'].abs() >= log2fc_cutoff)]
    top_genes = sig_genes.nlargest(top_n, '-log10pvals_adj')

    # Create figure with secondary x-axis for gene labels
    fig = go.Figure()

    # Add scatter plot of non-significant genes
    fig.add_trace(go.Scatter(
        x=de_results['log2foldchanges'],
        y=de_results['-log10pvals_adj'],
        mode='markers',
        name='not DE',
        marker=dict(color='grey', size=5),
        text=de_results['gene_names'],
        hoverinfo='text' ))
    
    for i, row in top_genes.iterrows():
        fig.add_annotation(
            x=row['log2foldchanges'],
            y=row['-log10pvals_adj'],
            text=row['gene_names'],
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='black')
        

    # Add scatter plot of significant genes
    fig.add_trace(go.Scatter(
        x=sig_genes['log2foldchanges'],
        y=sig_genes['-log10pvals_adj'],
        mode='markers',
        name='DE',
        marker=dict(color='red', size=5),
        text=sig_genes['gene_names'],
        hoverinfo='text'))

    # Add labels for the top N significant genes
    for i, row in top_genes.iterrows():
        fig.add_annotation(
            x=row['log2foldchanges'],
            y=row['-log10pvals_adj'],
            text=row['gene_names'],
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='black')

    # Set layout properties
    fig.update_layout(
        title=f'Interactive Volcano Plot ',
        xaxis_title='log2 Fold Change',
        yaxis_title='-log10 Adjusted P-value',
        template='plotly_white', width=1000,  # Set the width of the plot
    height=600,  # Set the height of the plot
    xaxis=dict(range=[-7, 7]),
    yaxis=dict(range=[0, 15]) )

    # Add horizontal line for p-value cutoff
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=-max(de_results['log2foldchanges']), x1=max(de_results['log2foldchanges']),
        y0=-np.log10(pval_cutoff), y1=-np.log10(pval_cutoff))

    # Add vertical lines for log2fc cutoffs
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=-log2fc_cutoff, x1=-log2fc_cutoff,
        y0=0, y1=max(de_results['-log10pvals_adj'])  )
    fig.add_shape(
        type='line',
        line=dict(dash='dash', color='black', width=1),
        x0=log2fc_cutoff, x1=log2fc_cutoff,
        y0=0, y1=max(de_results['-log10pvals_adj']))

    # Show figure
    fig.show()


def plot_volcano(de_results,  pval_cutoff=0.05, log2fc_cutoff=1):
    
    

    # Replace zeros and negative values in 'logfoldchanges' to avoid log2 of non-positive numbers
    de_results['logfoldchanges'] = de_results['logfoldchanges'].replace(0, np.nextafter(0, 1))
    de_results['log2foldchanges'] = np.sign(de_results['logfoldchanges']) * np.log2(np.abs(de_results['logfoldchanges']))

    # Calculate the -log10 of adjusted p-values
    de_results['-log10pvals_adj'] = -np.log10(de_results['pvals_adj'].replace(0, np.nextafter(0, 1)))
    top_genes = de_results.nlargest(10, '-log10pvals_adj')

    # Plotting with Plotly
    fig = px.scatter(
        de_results, 
        x='log2foldchanges', 
        y='-log10pvals_adj',
        color=np.where((de_results['pvals_adj'] < pval_cutoff) & (np.abs(de_results['log2foldchanges']) >= log2fc_cutoff), 'DE', 'not DE'),
        hover_data=['gene_names']
    )

    for i, row in top_genes.iterrows():
        fig.add_annotation(
            x=row['log2foldchanges'],
            y=row['-log10pvals_adj'],
            text=row['gene_names'],
            showarrow=True,
            arrowhead=1,
            arrowsize=2,
            arrowwidth=1,
            arrowcolor='black')
        
    # Update traces to customize marker size
    fig.update_traces(marker=dict(size=10))

    # Add horizontal line for significance threshold
    fig.add_hline(y=-np.log10(pval_cutoff), line_dash="dash", line_color="black")

    # Add vertical lines for log2 fold change threshold
    fig.add_vline(x=log2fc_cutoff, line_dash="dash", line_color="black")
    fig.add_vline(x=-log2fc_cutoff, line_dash="dash", line_color="black")

    # Update layout with titles and legend
    fig.update_layout(
        title='Volcano Plot of Differential Gene Expression',
        xaxis_title='log2 fold change',
        yaxis_title='-log10 adjusted p-value',
        legend_title='Significance',
        showlegend=True,     
        width=800,  # Set the width of the plot
        height=600,  # Set the height of the plot
        xaxis=dict(range=[-7, 7]),
        yaxis=dict(range=[0, 15])
    )

    # Show plot
    
    
    
def main():
    adata, results_file = load_and_preprocess_data()
    adata = filter_and_qc(adata)
    print("\n")
    adata = preprocess_data(adata)
    print("\n")
    adata = run_clustering_and_embedding(adata)
    print("\n")
    adata = run_differential_expression_analysis(adata)
    print("\n")

  
    groups = adata.obs['leiden'].unique().tolist()
    print("\n")
    
    type(groups)
    print(groups)
    
    create_i_volcano_plot(adata,groups)
    print("\n")
    print("\n")
    
    de_results = extract_de_results(adata, groups)

    plot_volcano(de_results)
        
if __name__ == "__main__":
    main()

