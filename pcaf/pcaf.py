import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #PCA tool
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import mahalanobis
import time

import bokeh
from bokeh.layouts import gridplot
from bokeh.plotting import output_notebook
from random import random

from bokeh.io import output_notebook  # prevent opening separate tab with graph
from bokeh.io import show

from bokeh.layouts import row
from bokeh.layouts import grid
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.models import Button  # for saving data
from bokeh.events import ButtonClick  # for saving data
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from bokeh.models import HoverTool
from bokeh.plotting import figure
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def pca(data):
    """ generates a PCA object and the pca_data for graphing. 
    Use like: pca_object, pca_data = pca(data)

    Parameters
    ----------
    data: array-like
        array of sample data for the pca.
    
    Returns
    -------
    PCA object
    Dataframe of samples projected onto princple components. Samples as rows, principle components as columns
    
    """
    pca = PCA()
    pca.fit(data)
    pca_data = pca.transform(data)
    pca_df = pd.DataFrame(pca_data)
    pca_df.columns = ['PC' + str(x) for x in range(1,len(pca_df.columns)+1)]
    return pca,pca_df


def eleLoads(pca, icpms_df, n_components='all'):
    """Returns a dataframe of the elemental loadings for each principle component or the number of PC's passed in the argument.
    Parameters
    ----------
    pca : pca.fit() object
    
    icpms_df : pandas dataframe object
        original dataframe of icpms data to generate the element labels

    n_components : int, optional
        default is 'all'. When an int is passed it will generate that number of principle components instead

    Returns
    -------
    pandas dataframe object
        with princple components as columns and elements as rows"""\

    if n_components != 'all':
        element_loads = pd.DataFrame(pca.components_[0:n_components])    
    else:
        element_loads = pd.DataFrame(pca.components_)
        
    element_loads.columns = icpms_df.columns
    element_loads.index = ['PC' + str(x) for x in range(1,len(element_loads.index)+1)]
    element_loads = element_loads.transpose()
    return element_loads



def pc_variance(pca):
    """ This function produces an arrray of principle component variance values as percentages 
    also returns a list of labels equal to the number of princple components. 
    use like: variance, labels = pc_labels(pca)
    
    Parameters
    ----------
    pca: PCA class object
    
    Returns
    -------
    variance_df: Pandas DataFrame object with two columns
    'PC' containing Principle component numbers and 'Variance' containing variance explained
    """
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=4) # variances rounded out and made an array
    labels = ['PC' + str(x) for x in range(1,len(per_var)+1)] #labels equal to the number of PCs
    dict_combo = {'PC':labels,'Variance':per_var}
    variance_df = pd.DataFrame(dict_combo)
    return variance_df

def quick_scree(pca, style='bar', figsize=(25,10)):
    """ Returns a basic scree plot of the pca data passed into it. Good for quick visualization
    
    Parameters
    ----------
    pca: PCA class object
    
    style: str
        'bar' or 'line'
    
    figsize: tuple
        tuple of two values for the figure size. Default is (25,10)
    
    """
    if style == 'bar':
        plot_style = plt.bar
    elif style == 'line':
        plot_style = plt.plot
    
    variance_df = pc_variance(pca)
    plt.figure(figsize=figsize)
    plot_style(variance_df['PC'], variance_df['Variance'])
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principle Component')
    plt.title('Scree Plot')
    plt.show()

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

   # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
def scatter(pca_df,x='PC1', y='PC2', title="Biplot", label_column=None, color_column=None, conf_ellipse=None, figsize=(15,15)):
    """Creates a basic biplot based on the passed in PC numbers. 
    Essentially creates a scatter plot from two columns in a dataframe lol.
    
    Parameters
    ----------
    pca_df: pandas dataframe
        
    x: string
        default "PC1" - select which principle component to include on the x axis or which column
        
    y: string
        default "PC2" - select which principle component to include on the y axis or which column
    
    title: string
        default "Biplot"
    
    label_column: string
        default None. Column name for labels. use "index" to label by the dataframe index
    
    color_column: string
        default None. Column used for color categories. Use 'index' to color by the dataframe index
    
    conf_ellipse: string
        default None. Column used for confidence ellipses. For use with defined groups
    
    figsize: tuple
        tuple of two values for the figure size. Default is (15,15)
    
    Returns
    -------
    
    Scatter plot showing the passed in arguments
    
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_column == None:
        
        ax.scatter(x, y,data=pca_df)
    elif color_column == 'index':
        ax.scatter(x, x, c= pca_df.index, cmap='Set2',data=pca_df)
    else:
        ax.scatter(x, y, c = color_column, cmap='Set2',data=pca_df)
    
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    if label_column == None:
        pass
    elif label_column == 'index':
        for sample in pca_df.index:
            ax.annotate(sample, (pca_df[x].loc[sample], pca_df[y].loc[sample]))
        
    else:
        for sample in pca_df.index:
            ax.annotate(str(pca_df.loc[sample,label_column]), (pca_df[x].loc[sample], pca_df[y].loc[sample]))
    
    if conf_ellipse == None:
        pass
    else:
        for i in pca_df[conf_ellipse].unique():
            confidence_ellipse(pca_df.loc[pca_df[conf_ellipse]==i, x], pca_df.loc[pca_df[conf_ellipse]==i, y], ax, edgecolor='red')
    
                 
    plt.show();



def ox_to_ppm(df, oxides):
    """
    transforms columns in a database from oxides to ppm
    
    Parameters
    ----------
    
    df: pandas dataframe
    
    oxides: list-like
        list of columns in the dataframe to convert to ppm
        
    Returns
    -------
    
    A copy of the passed dataframe with the indicated columns converted to ppm
    
    """
    df_copy = df.copy()
    for ox in oxides:
        df_copy[ox] = df_copy[ox]*10000
    return df_copy

def quick_dendro(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def othercrit(p,n,a):
    return ((n*p*(n-2))/((n-1)*(n-p-1)))*f.isf((a/n), p, (n-p-1))


def md_chart(df, group_col, p_val=0.05):
    
    
    #Pull out just the measurement columns
    measurement_columns = list(df.columns)
    measurement_columns.remove(group_col)
    result_df = pd.DataFrame(index=df.index, columns=['assigned_group','suggested_group'])
    groups_columns = []
    for g in df[group_col].unique():
        result_df['mhd_gp'+str(g)]=np.nan
        result_df['in_gp'+str(g)]=np.nan
        groupy = 'in_gp'+str(g)
        groups_columns.append(groupy) #for use later when suggesting a group
    
    #This is a for loop to get through all the samples in the input dataframe
    for sample in df.index:
        
        #This is for jacknife mahalanobis (leave one out)
        sample_only = df.loc[df.index == sample, measurement_columns]
        #everything that is not the sample to be distanced. Leaving in the group column to use it later
        non_sample = df.loc[df.index != sample,:]
        result_df.loc[result_df.index==sample, 'assigned_group'] = df.loc[df.index==sample, group_col]
        
        #iterating through all the groups provided
        for g in df[group_col].unique():
            
            #The meat of the mahalanobis stuff
            group = non_sample.loc[non_sample[group_col]==g, measurement_columns] #dropping the group column
            group_mean = group.mean() #finding the group mean
            
            #using a psuedo-inverse covariance matrix, this is used for groups with small n.
            group_psuedo_inverse_covariance = np.linalg.pinv(group.cov())
            
            #using scipy mahalanobis distance measure function
            mhd = mahalanobis(sample_only, group_mean, group_psuedo_inverse_covariance)
            
            #using the Wilks method to generate critical values
            n = len(group)+1
            p = len(group.columns)
            g_col = 'mhd_gp'+str(g)
            g_crit_col = 'in_gp'+str(g)
            
            result_df.loc[result_df.index==sample,g_col] = mhd**2
            result_df.loc[result_df.index==sample,g_crit_col] = round((f.sf((((n-p-1)/(p*(n-2)))*(mhd**2)),p,(n-p-1))*100),5)
        
        max_group = result_df.loc[result_df.index==sample, groups_columns].idxmax(axis=1)
        
        max_val = result_df.loc[result_df.index==sample, [str(max_group.iloc[0])]]
        
        if float(int(max_val.iloc[0])) >0.001:
            val = str(max_group.iloc[0])
            val = int(val[-1])
            result_df.loc[result_df.index==sample, 'suggested_group'] = val
        else:
            
            result_df.loc[result_df.index==sample, 'suggested_group'] = 'All group probabilities are < 0.001 '
    
    
    return(result_df)

def point_selection(dataframe, x_axis, y_axis, label="index", plot_width=500, plot_height=500, file_name='selection'):
    """
    Outputs a scatterplot from the passed dataframe and columns. Points on the scatter plot can be selected and exported to a seperate csv file.
    
    Parameters
    ----------
    
    dataframe: pandas dataframe object
    
    x_axis: string
        string indicating which column from the dataframe to be on the x-axis
    
    y_axis: string
        string indicating which column from the dataframe to be on the y-axis
    label: string
        string indicating which column (or the index) of the dataframe for the label column in the exported csv. Default = 'index' which results in the index numbers for the points
    
    plot_width: int
        indicate plot width in pixels. Default = 500
    
    plot_height: int
        indicate plot height in pixels. Default = 500
    
    file_name: string
        string with the desired file name. Function will add its own file extension. default is "selection" (results in "selection.txt")
    
    Returns
    -------
    Interactive scatter plot with a button to save selected points as a csv
    
    """
    
    file_export = file_name + '.txt'
    s1 = ColumnDataSource(dataframe)
    fig01 = figure(
        plot_width=plot_width,
        plot_height=plot_height,
        tools=["lasso_select", "reset", "save"],
        title="Select Here",)

    fig01.circle(x_axis, y_axis, source=s1, alpha=0.6)

    # create second subplot
    s2 = ColumnDataSource(data=dict(x=[], y=[]))

    # demo smart error msg:  `box_zoom`, vs `BoxZoomTool`
    fig02 = figure(
        plot_width=500,
        plot_height=500,
        x_range=fig01.x_range,
        y_range=fig01.y_range,
        tools=["box_zoom", "wheel_zoom", "reset", "save"],
        title="Watch Here",
    )

    fig02.circle("x", "y", source=s2, alpha=0.6, color="firebrick")

    # create dynamic table of selected points
    columns = [
        TableColumn(field="x", title=y_axis),
        TableColumn(field="y", title=x_axis),
        TableColumn(field="label", title=label)
    ]

    table = DataTable(
        source=s2,
        columns=columns,
        width=400,
        height=600,
        sortable=True,
        selectable=True,
        editable=True,
    )

    # fancy javascript to link subplots
    # js pushes selected points into ColumnDataSource of 2nd plot
    # inspiration for this from a few sources:
    # credit: https://stackoverflow.com/users/1097752/iolsmit via: https://stackoverflow.com/questions/48982260/bokeh-lasso-select-to-table-update
    # credit: https://stackoverflow.com/users/8412027/joris via: https://stackoverflow.com/questions/34164587/get-selected-data-contained-within-box-select-tool-in-bokeh

    s1.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(s1=s1, s2=s2, table=table, x_axis=x_axis, y_axis=y_axis, label=label),
            code="""
            var inds = cb_obj.indices;
            var d1 = s1.data;
            var d2 = s2.data;
            d2['x'] = []
            d2['y'] = []
            d2['label'] = []
            for (var i = 0; i < inds.length; i++) {
                d2['x'].push(d1[x_axis][inds[i]])
                d2['y'].push(d1[y_axis][inds[i]])
                d2['label'].push(d1[label][inds[i]])
            }
            s2.change.emit();
            table.change.emit();
            testing.change.emit();

            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = "x, y\\n";
            for (i = 0; i < inds.length; i++) {
                out += data['x'][inds[i]] + "," + data['y'][inds[i]] + "\\n";
            }
            var file = new Blob([out], {type: 'text/plain'});

        """,
        ),
    )

    # create save button - saves selected datapoints to text file onbutton
    # inspriation for this code:
    # credit:  https://stackoverflow.com/questions/31824124/is-there-a-way-to-save-bokeh-data-table-content
    # note: savebutton line `var out = "x, y\\n";` defines the header of the exported file, helpful to have a header for downstream processing

    savebutton = Button(label="Save", button_type="success")
    savebutton.js_on_event(ButtonClick, CustomJS(
        args=dict(source_data=s1, file_export = file_export,  x_axis=x_axis, y_axis=y_axis, label=label),
        code="""
            var inds = source_data.selected.indices;
            var data = source_data.data;
            var out = label + "," + x_axis +","+y_axis+"\\n";
            for (var i = 0; i < inds.length; i++) {
                out += data[label][inds[i]] + "," + data[x_axis][inds[i]] + "," + data[y_axis][inds[i]] + "\\n";
            }
            var file = new Blob([out], {type: 'text/plain'});
            var elem = window.document.createElement('a');
            elem.href = window.URL.createObjectURL(file);
            elem.download = file_export;
            document.body.appendChild(elem);
            elem.click();
            document.body.removeChild(elem);
            """
            )
                             )

    # add Hover tool
    # define what is displayed in the tooltip
    tooltips = [
        ("X:", "@x"),
        ("Y:", "@y"),
        ("static text", "static text"),
    ]

    fig02.add_tools(HoverTool(tooltips=tooltips))

    # display results
    # demo linked plots
    # demo zooms and reset
    # demo hover tool
    # demo table
    # demo save selected results to file

    layout = grid([fig01, fig02, table, savebutton], ncols=4)

    output_notebook()
    show(layout)
