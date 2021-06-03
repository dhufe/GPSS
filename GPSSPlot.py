import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import pybamcmap 

class GPSSPlot:
    @staticmethod 
    def set_style():
        # This sets reasonable defaults for font size for
        # a figure that will go in a paper
        sns.set_context("paper")
    
        # Set the font to be serif, rather than sans
        sns.set(font='serif',style="ticks")

        # Make the background white, and specify the
        # specific font family
        sns.set_style("white", {
            "font.family": "serif",
            "font.serif": ["Times", "Palatino", "serif"]
        })
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    @staticmethod 
    def set_size(fig, width, height ):
        fig.set_size_inches(6, 4)
        fig.tight_layout()


    @staticmethod 
    def PlottingSourceConfiguration ( Xs, Ys, Zs, Ps = None, fileName = 'PSS_Source_configuration.png' ):
        """
    
        """
        if Ps == None:
            MarkerSize = 1
        else:
            MarkerSize = Ps 

        GPSSPlot.set_style()

        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter ( Xs*1e3, Ys*1e3, s=MarkerSize, c='r', marker='o')
        ax[0][0].set_xlabel ( r'$x$ / $mm$' )
        ax[0][0].set_ylabel ( r'$y$ / $mm$' )
        ax[0][0].grid(True)
    
        ax[0][1].scatter ( Xs*1e3, Zs*1e3, s=MarkerSize, c='r', marker='o')
        ax[0][1].set_xlabel ( r'$x$ / $mm$' )
        ax[0][1].set_ylabel ( r'$z$ / $mm$' )
        ax[0][1].grid(True)

        ax[1][0].scatter ( Ys*1e3, Zs*1e3, s=MarkerSize, c='r', marker='o')
        ax[1][0].set_xlabel ( r'$y$ / $mm$' )
        ax[1][0].set_ylabel ( r'$z$ / $mm$' )
        ax[1][0].grid(True)

        GPSSPlot.set_size(fig, 6, 6/1.68)
        plt.savefig( fileName, dpi=300 )
        plt.close(fig)

    @staticmethod 
    def PlotFieldData ( fileName, data, Xmesh, Ymesh ):
        h, w = data.shape
        data_ratio = h / float(w)

        ### axes lengths in inche
        width_ax0 = 8.
        width_ax1 = 2.
        height_ax2 = 2.

        Xmesh *= 1e3
        Ymesh *= 1e3

        height_ax0 = width_ax0 * data_ratio

        ### define margins size in inche
        left_margin  = 1.00
        right_margin = 0.2
        bottom_margin = 0.5
        top_margin = 0.25
        inter_margin = 0.5

        ### calculate total figure size in inche
        fwidth = left_margin + right_margin + inter_margin + width_ax0 + width_ax1
        fheight = bottom_margin + top_margin + inter_margin + height_ax0 + height_ax2

        fig = plt.figure(figsize=(fwidth, fheight))
        fig.patch.set_facecolor('white')

        ### create axes
        ax1 = fig.add_axes([(left_margin ) / fwidth,
                        (bottom_margin + inter_margin + height_ax2) / fheight,
                         width_ax1 / fwidth, height_ax0 / fheight])

        ax0 = fig.add_axes([(left_margin + width_ax1 + inter_margin) / fwidth,
                        (bottom_margin + inter_margin + height_ax2) / fheight,
                        width_ax0 / fwidth, height_ax0 / fheight])

        ax2 = fig.add_axes([(left_margin + width_ax1 + inter_margin) / fwidth, bottom_margin / fheight,
                        width_ax0 / fwidth, height_ax2 / fheight])

        ### plotting the data
        bounds = [ Xmesh.min(),Xmesh.max(),Ymesh.min(),Ymesh.max()]
        GPSSPlot.set_style()
   
        im1 = ax0.pcolor ( Xmesh, Ymesh, data, cmap='bam_jet' ) 

        line = data[:, w//2]

        ax1.plot( line , Ymesh[:,w//2], 'k-', linewidth=1.0 )
        ax1.set_ylim ( bounds[2], bounds[3] )
        ax1.set_xlim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
        ax1.invert_xaxis()

        idx_max_x = int(np.argmax(line)) 
        line = data[ idx_max_x , : ]
        ax2.plot( Xmesh[ idx_max_x, : ], line , 'k-', linewidth=1.0 )
        ax2.set_xlim ( bounds[0], bounds[1] )
        ax2.set_ylim ( .9 * np.amin (line) , 1.1 * np.amax (line) )
        plt.savefig( fileName + '.png', dpi=300 )
        plt.savefig( fileName + '.pdf', dpi=300 )
        plt.close(fig)

