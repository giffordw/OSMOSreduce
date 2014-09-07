'''Build like you want to distribute on GitHub'''
import numpy as np
import matplotlib.pyplot as plt

class DragSpectra:
    def __init__(self,spectra,ydata,ax):
        self.ax = ax
        print 'begin shift'
        self.spectra = spectra
        self.ydata = ydata
        self.pressed = False
        self.dx_tot = 0.0
        #figure.canvas.mpl_connect('motion_notify_event',self.on_motion)
        #figure.canvas.mpl_connect('button_press_event',self.on_press)
        #figure.canvas.mpl_connect('button_release_event',self.on_release)

    def on_motion(self,evt):
        if self.pressed:
            dx = evt.xdata - self.mouse_x
            #self.dx_tot += dx
            self.spectra.set_data(self.spectra_x + dx,self.ydata)
            plt.draw()

    def on_press(self,evt):
        if evt.inaxes == self.ax:
            self.mouse_x = evt.xdata
            self.spectra_x = self.spectra.get_xdata()
            self.pressed = True
            #print 'mouse press'
        else: return

    def on_release(self,evt):
        if evt.inaxes == self.ax:
            self.pressed = False
            #print evt.xdata - self.mouse_x
            self.dx_tot += evt.xdata - self.mouse_x
            print self.dx_tot
            #print 'release event'
        else: return
