#TESTING PLOTTING:


######################################### STUFF I PICKED OUT AN BATCHED ########################

###### HOW TO PLOT THE FASTEST WAY ########
import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
from pyqtgraph.ptime import time    
##widget/axes itself within the window
graphics_layout_axes = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
#setup window into which we will insert a axes/widget:
app = QtGui.QApplication([])
gui_main_window = QtGui.QMainWindow()
gui_main_window.resize(600,600)
gui_main_window.setCentralWidget(graphics_layout_axes)
gui_main_window.setWindowTitle('pyqtgraph example: ScatterPlot')
gui_main_window.show()
#add plot to axes/widget:
plot_handle = graphics_layout_axes.addPlot()
#Get Points to Scatter:
spots = []; #initialize
bla1 = tuple(np.random.randn(2)+100);
spots.append(  {'pos': bla1, 'size': 1e1, 'pen': {'color': 'w', 'width': 2} }  );
#within the plot insert a scatter plot item (size=10?????):
scatter_plot_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255,255,255,120));
scatter_plot_item.addPoints(spots);
plot_handle.addItem(scatter_plot_item);
#to clear dots:
plot_handle.clear();



win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')
p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)
p6 = win.addPlot(title="Updating plot")
p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
p7.plot(y, fillLevel=-0.3, brush=(50,50,200,100))
p7.showAxis('bottom', False)
win.nextRow()
p7 = win.addPlot(title="Filled plot, axis disabled")
p9.getViewBox().viewRange()


gui_graphics_window = pg.GraphicsWindow(title="Basic plotting examples")
gui_graphics_window.resize(1000,600)
gui_graphics_window.setWindowTitle('pyqtgraph example: Plotting')
p5 = gui_graphics_window.addPlot(title="Scatter plot, axis labels, log scale")
x = np.random.normal(size=1000) * 1e-5
y = x*1000 + 0.005 * np.random.normal(size=1000)
y -= y.min()-1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
# FULL CONTROL OVER PLOT STYLE!!!!!!:
# HERE, for a pg.GraphicWindow().addPlot() item it seems i need to use .clear() to clear the plot!!!!
# ^ i can't seem to find a way to s
p5.plot(x, y+1, fillLevel = -0.3, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)
p5.showAxis('bottom', False)
p5.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
#Highlight region in the plot within the X-axs range determined below (between 400 and 700 here):
linear_region_x_axis_delimiter = pg.LinearRegionItem([400,700])
linear_region_x_axis_delimiter.setZValue(-10)
p5.addItem(linear_region_x_axis_delimiter)
#Add an axis to the right for multiple axes:
p5.showAxis('right')
#SET AXES LIMITS!!!:
p5.setXRange(5, 20, padding=0)
p5.setYRange(5, 20, padding=0)
#MAYBE THIS IS HOW I CHANGE THE DATA WITHOUT USING THE .clear() FUNCTION:
p5.plotItem.plot().setData()




pg.mkQApp()
pw = pg.PlotWidget()
pw.show()
pw.setWindowTitle('pyqtgraph example: MultiplePlotAxes')
p1 = pw.plotItem
p1.setLabels(left='axis 1')
## create a new ViewBox, link the right axis to its coordinate system
p2 = pg.ViewBox()
p1.showAxis('right')
p1.scene().addItem(p2)
p1.getAxis('right').linkToView(p2)
p2.setXLink(p1)
p1.getAxis('right').setLabel('axis2', color='#0000ff')
p1.plot([1,2,4,8,16,32])
p2.addItem(pg.PlotCurveItem([10,20,40,80,40,20], pen='b'))
p3.addItem(pg.PlotCurveItem([3200,1600,800,400,200,100], pen='r'))


app = QtGui.QApplication([])
p = pg.plot()
p.setWindowTitle('pyqtgraph example: PlotSpeedTest')
p.setRange(QtCore.QRectF(0, -10, 5000, 20)) 
p.setLabel('bottom', 'Index', units='B')
curve = p.plot()          
data = np.random.normal(size=(50,5000))
ptr = 0
curve.setData(data[ptr%10])
ptr += 1
p.setTitle('%0.2f fps' % fps)
app.processEvents()  ## force complete redraw for every plot
    

app = pg.mkQApp()
plt = pg.PlotWidget()
app.processEvents()
## Putting this at the beginning or end does not have much effect
plt.show()   
## The auto-range is recomputed after each item is added,
## so disabling it before plotting helps
plt.enableAutoRange(False, False)    


p1 = win.addPlot(title="95th percentile range", y=d)
p1.enableAutoRange('y', 0.95)
p2 = win.addPlot(title="Auto Pan Only")
p2.setAutoPan(y=True)



## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(800,800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')
text = """Data Selection From Image.<br>\n
Drag an ROI or its handles to update the selected image.<br>
Hold CTRL while dragging to snap to pixel boundaries<br>
and 15-degree rotation angles.
"""
w1 = w.addLayout(row=0, col=0)
label1 = w1.addLabel(text, row=0, col=0)
v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
v1b = w1.addViewBox(row=2, col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
img1b = pg.ImageItem()
v1b.addItem(img1b)
v1a.disableAutoRange('xy')
v1b.disableAutoRange('xy')
v1a.autoRange()
v1b.autoRange()
#
rois = []
rois.append(pg.RectROI([20, 20], [20, 20], pen=(0,9)))
rois[-1].addRotateHandle([1,0], [0.5, 0.5])
rois.append(pg.LineROI([0, 60], [20, 80], width=5, pen=(1,9)))
rois.append(pg.MultiRectROI([[20, 90], [50, 60], [60, 90]], width=5, pen=(2,9)))
rois.append(pg.EllipseROI([60, 10], [30, 20], pen=(3,9)))
rois.append(pg.CircleROI([80, 50], [20, 20], pen=(4,9)))

w3 = w.addLayout(row=1, col=0)
label3 = w3.addLabel(text, row=0, col=0)
v3 = w3.addViewBox(row=1, col=0, lockAspect=True)
r3a = pg.ROI([0,0], [10,10])
v3.addItem(r3a)
## handles scaling horizontally around center
r3a.addScaleHandle([1, 0.5], [0.5, 0.5])
r3a.addScaleHandle([0, 0.5], [0.5, 0.5])

#SET AXIS RANGE:
p1.setXRange(5, 20, padding=0)

##########################################################################################################









######################################### SIMPLE PLOTTING #######################################
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

#Simplest, Quickest plotting method. I SHOULD IMPORT IT AS pgplot !!!!!!
bla = pg.plot(np.random.normal(size=100000), title="Simplest possible plotting example")

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()


######################################### SCATTER PLOTS #######################################
## Add path to library (just for examples; you do not need this)

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

gui_app = QtGui.QApplication([])

#Most general outer container - the gui_main_window (kind of like "parent" in matlab):
#To a gui_main_window i CANNOT do a .addPlot() , i need to add an axes.
gui_main_window = QtGui.QMainWindow()
gui_main_window.resize(600,600)

#Within the gui_main_window i put an axes 
graphics_layout_axes = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
gui_main_window.setCentralWidget(graphics_layout_axes)

#Then i tell the gui_main_window to be visible...even if i click X it's still running in the background ready
#to be shown:
gui_main_window.show()
gui_main_window.setWindowTitle('pyqtgraph example: ScatterPlot')

## create four areas to add plots
subplot_1 = graphics_layout_axes.addPlot()
subplot_2 = graphics_layout_axes.addViewBox()
subplot_2.setAspectLocked(True)
#### 
#####   NEXT ROW IN THE AXES (LIKE SUBPLOT(2,2,1)):
graphics_layout_axes.nextRow()
#####
subplot_3 = graphics_layout_axes.addPlot()
subplot_4 = graphics_layout_axes.addPlot()
print("Generating data, this takes a few seconds...")

## There are a few different ways we can draw scatter plots; each is optimized for different types of data:


## 1) All spots identical and transform-invariant (top-left plot). 
## In this case we can get a huge performance boost by pre-rendering the spot 
## image and just drawing that image repeatedly.

n = 300
scatter_plot_item_1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
positions_xy = np.random.normal(size=(2,n), scale=1e-5)
scatter_spots_list_in_right_format = [{'pos': positions_xy[:,i], 'data': 1} for i in range(n)] + [{'pos': [0,0], 'data': 1}]
scatter_plot_item_1.addPoints(scatter_spots_list_in_right_format)
subplot_1.addItem(scatter_plot_item_1)


#####
#IF I WANT TO INTERACTIVELY GET DATAPOINTS CONTENT PRINTED IN IPYTHON:
## Make all plots clickable
lastClicked = []
def clicked(plot, points):
    global lastClicked
    for p in lastClicked:
        p.resetPen()
    print("clicked points", points)
    for p in points:
        p.setPen('b', width=2)
    lastClicked = points
scatter_plot_item_1.sigClicked.connect(clicked)
#####


## 2) Spots are transform-invariant, but not identical (top-right plot). 
## In this case, drawing is as fast as 1), but there is more startup overhead
## and memory usage since each spot generates its own pre-rendered image.

scatter_plot_item_2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen('w'), pxMode=True)
positions_xy = np.random.normal(size=(2,n), scale=1e-5)
scatter_spots_list_in_right_format = [{'pos': positions_xy[:,i], 'data': 1, 'brush':pg.intColor(i, n), 'symbol': i%5, 'size': 5+i/10.} for i in range(n)]
scatter_plot_item_2.addPoints(scatter_spots_list_in_right_format)
subplot_2.addItem(scatter_plot_item_2)
scatter_plot_item_2.sigClicked.connect(clicked)


## 3) Spots are not transform-invariant, not identical (bottom-left). 
## This is the slowest case, since all spots must be completely re-drawn 
## every time because their apparent transformation may have changed.

scatter_plot_item_3 = pg.ScatterPlotItem(pxMode=False)   ## Set pxMode=False to allow spots to transform with the view
scatter_spots_list_in_right_format = []
for i in range(10):
    for j in range(10):
        scatter_spots_list_in_right_format.append({'pos': (1e-6*i, 1e-6*j), 'size': 1e-6, 'pen': {'color': 'w', 'width': 2}, 'brush':pg.intColor(i*10+j, 100)})
scatter_plot_item_3.addPoints(scatter_spots_list_in_right_format)
subplot_3.addItem(scatter_plot_item_3)
scatter_plot_item_3.sigClicked.connect(clicked)


## Test performance of large scatterplots

scatter_plot_item_4 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 20))
positions_xy = np.random.normal(size=(2,10000), scale=1e-9)
scatter_plot_item_4.addPoints(x=positions_xy[0], y=positions_xy[1])
subplot_4.addItem(scatter_plot_item_4)
scatter_plot_item_4.sigClicked.connect(clicked)



#SHOULD I ADD THIS INTO MY SCRIPT???
## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()





######################################### 2D PLOTTING PLOTS #######################################
#This example demonstrates many of the 2D plotting capabilities
#in pyqtgraph. All of the plots may be panned/scaled by dragging with 
#the left/right mouse buttons. Right click on any plot to show a context menu.

import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
gui_app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

#Looks like this is the same as the gui_main_window from before only now it's called graphics_window:
#The main difference is that a graphics_window apparently is more geared towards plotting and graphics - 
#so i can directly use the .addPlot() function. 
gui_graphics_window = pg.GraphicsWindow(title="Basic plotting examples")
gui_graphics_window.resize(1000,600)
gui_graphics_window.setWindowTitle('pyqtgraph example: Plotting')

#I add into the graphics window a plot
#plot 1:
p1 = gui_graphics_window.addPlot(title="Basic array plotting", y=np.random.normal(size=100))
#plot 2 (TO THE RIGHT OF plot1):
p2 = gui_graphics_window.addPlot(title="Multiple curves")

#Now i fill plot2 - notice i don't have to use a "hold on;" or something...it just adds the wanted points:
#(how do i clear points or focus on a specific part of the y or x axes? like xlim and ylim)
p2.plot(np.random.normal(size=100), pen=(255,0,0))
p2.plot(np.random.normal(size=100)+5, pen=(0,255,0))
p2.plot(np.random.normal(size=100)+10, pen=(0,0,255))

#Anoher plot to the right of plot2 (i still didn't use .nextrow()!!!!):
p3 = gui_graphics_window.addPlot(title="Drawing with points")
p3.plot(np.random.normal(size=100), pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')

#go down a row:
gui_graphics_window.nextRow()


p4 = gui_graphics_window.addPlot(title="Parametric, grid enabled")
x = np.cos(np.linspace(0, 2*np.pi, 1000))
y = np.sin(np.linspace(0, 4*np.pi, 1000))
p4.plot(x, y)
p4.showGrid(x=True, y=True)

p5 = gui_graphics_window.addPlot(title="Scatter plot, axis labels, log scale")
x = np.random.normal(size=1000) * 1e-5
y = x*1000 + 0.005 * np.random.normal(size=1000)
y -= y.min()-1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
# FULL CONTROL OVER PLOT STYLE!!!!!!:
p5.plot(x, y, fillLevel = -0.3, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)
p5.showAxis('bottom', False)
p5.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
#Highlight region in the plot within the X-axs range determined below (between 400 and 700 here):
linear_region_x_axis_delimiter = pg.LinearRegionItem([400,700])
linear_region_x_axis_delimiter.setZValue(-10)
p5.addItem(linear_region_x_axis_delimiter)
#Add an axis to the right for multiple axes:
p5.showAxis('right')
#SET AXES LIMITS!!!:
p5.setXRange(5, 20, padding=0)
p5.setYRange(5, 20, padding=0)


p6 = gui_graphics_window.addPlot(title="Updating plot")
curve = p6.plot(pen='y')
data = np.random.normal(size=(10,1000))
ptr = 0
def update():
    global curve, data, ptr, p6
    curve.setData(data[ptr%10])
    if ptr == 0:
        p6.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1

#how do i set timer time????
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


gui_graphics_window.nextRow()

p7 = gui_graphics_window.addPlot(title="Filled plot, axis disabled")
y = np.sin(np.linspace(0, 10, 1000)) + np.random.normal(size=1000, scale=0.1)
p7.plot(y, fillLevel=-0.3, brush=(50,50,200,100))
p7.showAxis('bottom', False)


x2 = np.linspace(-100, 100, 1000)
data2 = np.sin(x2) / x2
p8 = gui_graphics_window.addPlot(title="Region Selection")
p8.plot(data2, pen=(255,255,255,200))
lr = pg.LinearRegionItem([400,700])
lr.setZValue(-10)
p8.addItem(lr)

p9 = gui_graphics_window.addPlot(title="Zoom on selected region")
p9.plot(data2)
def updatePlot():
    p9.setXRange(*lr.getRegion(), padding=0)
def updateRegion():
    lr.setRegion(p9.getViewBox().viewRange()[0])
lr.sigRegionChanged.connect(updatePlot)
p9.sigXRangeChanged.connect(updateRegion)
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




######################################### MULTIPLE AXES PLOTS #######################################
#Demonstrates a way to put multiple axes around a single plot. 
#(This will eventually become a built-in feature of PlotItem)

import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


# mkQApp - untill now i have used  gui_app = QtGui.QApplication([]) , what's the difference?
pg.mkQApp()

# a PlotWidget - is this even a more direct way of plotting?
# is (again) widget here is like axes? like in graphics_layout_axes = pg.GraphicsLayoutWidget() ???
plot_axes = pg.PlotWidget()
plot_axes.show()
plot_axes.setWindowTitle('pyqtgraph example: MultiplePlotAxes')
# now instead of INSERTING a plot item , this axes already comes with a plot item attribute
p1 = plot_axes.plotItem
p1.setLabels(left='axis 1')

## create a new ViewBox, link the right axis to its coordinate system
# a ViewBox item?
p2 = pg.ViewBox()
p1.showAxis('right')
p1.scene().addItem(p2)
p1.getAxis('right').linkToView(p2)
p1.getAxis('right').setLabel('axis2', color='#0000ff')
p2.setXLink(p1)


## create third ViewBox. 
## this time we need to create a new axis as well.
p3 = pg.ViewBox()
ax3 = pg.AxisItem('right')
p1.layout.addItem(ax3, 2, 3)
p1.scene().addItem(p3)
ax3.linkToView(p3)
p3.setXLink(p1)
ax3.setZValue(-10000)
ax3.setLabel('axis 3', color='#ff0000')


## Handle view resizing 
def updateViews():
    ## view has resized; update auxiliary views to match
    global p1, p2, p3
    p2.setGeometry(p1.vb.sceneBoundingRect())
    p3.setGeometry(p1.vb.sceneBoundingRect())
    
    ## need to re-update linked axes since this was called
    ## incorrectly while views had different shapes.
    ## (probably this should be handled in ViewBox.resizeEvent)
    p2.linkedViewChanged(p1.vb, p2.XAxis)
    p3.linkedViewChanged(p1.vb, p3.XAxis)

updateViews()
p1.vb.sigResized.connect(updateViews)


p1.plot([1,2,4,8,16,32])
#To a ViewBox i add a pg.PlotCurveItem , just like i add a  ScatterPlotItem to a plot or a viewbox:
p2.addItem(pg.PlotCurveItem([10,20,40,80,40,20], pen='b'))
p3.addItem(pg.PlotCurveItem([3200,1600,800,400,200,100], pen='r'))

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()





######################################### UPDATE PLOT SPEED TEST #######################################
#Update a simple plot as rapidly as possible to measure speed.


## Add path to library (just for examples; you do not need this)
import initExample


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
app = QtGui.QApplication([])

#Quickest Way to Plot: pg.plot()
p = pg.plot()
p.setWindowTitle('pyqtgraph example: PlotSpeedTest')
p.setRange(QtCore.QRectF(0, -10, 5000, 20)); 
p.setLabel('bottom', 'Index', units='B')
curve = p.plot(); # pg.plot().plot()????? returns a curve item or someting?
data = np.random.normal(size=(50,5000))
curve.setData(data) #is there a .setData() function in the other examples? CHECK IT OUT!!!

#curve.setFillBrush((0, 0, 100, 100))
#curve.setFillLevel(0)

#lr = pg.LinearRegionItem([100, 4900])
#p.addItem(lr)

data = np.random.normal(size=(50,5000))
ptr = 0
lastTime = time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    curve.setData(data[ptr%10])
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




######################################### LOG PLOT #######################################
import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: LogPlotTest')


p5 = win.addPlot(title="Scatter plot, axis labels, log scale")
x = np.random.normal(size=1000) * 1e-5
y = x*1000 + 0.005 * np.random.normal(size=1000)
y -= y.min()-1.0
mask = x > 1e-15
x = x[mask]
y = y[mask]
p5.plot(x, y, pen=None, symbol='t', symbolPen=None, symbolSize=10, symbolBrush=(100, 100, 255, 50))
p5.setLabel('left', "Y Axis", units='A')
p5.setLabel('bottom', "Y Axis", units='s')
p5.setLogMode(x=True, y=False)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()





######################################### SCATTER PLOTS SPEED TEST #######################################
## Add path to library (just for examples; you do not need this)
import initExample


from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE
import numpy as np
import pyqtgraph as pg
from pyqtgraph.ptime import time
#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)
if USE_PYSIDE:
    from ScatterPlotSpeedTestTemplate_pyside import Ui_Form
else:
    from ScatterPlotSpeedTestTemplate_pyqt import Ui_Form

win = QtGui.QWidget()
win.setWindowTitle('pyqtgraph example: ScatterPlotSpeedTest')
ui = Ui_Form()
ui.setupUi(win)
win.show()

p = ui.plot

data = np.random.normal(size=(50,500), scale=100)
sizeArray = (np.random.random(500) * 20.).astype(int)
ptr = 0
lastTime = time()
fps = None
def update():
    global curve, data, ptr, p, lastTime, fps
    p.clear()
    if ui.randCheck.isChecked():
        size = sizeArray
    else:
        size = ui.sizeSpin.value()
    curve = pg.ScatterPlotItem(x=data[ptr%50], y=data[(ptr+1)%50], pen='w', brush='b', size=size, pxMode=ui.pixelModeCheck.isChecked())
    p.addItem(curve)
    ptr += 1
    now = time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    p.setTitle('%0.2f fps' % fps)
    p.repaint()
    #app.processEvents()  ## force complete redraw for every plot
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)
    


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()






######################################### CUSTOM PLOT #######################################
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import time

class DateAxis(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        strns = []
        rng = max(values)-min(values)
        #if rng < 120:
        #    return pg.AxisItem.tickStrings(self, values, scale, spacing)
        if rng < 3600*24:
            string = '%H:%M:%S'
            label1 = '%b %d -'
            label2 = ' %b %d, %Y'
        elif rng >= 3600*24 and rng < 3600*24*30:
            string = '%d'
            label1 = '%b - '
            label2 = '%b, %Y'
        elif rng >= 3600*24*30 and rng < 3600*24*30*24:
            string = '%b'
            label1 = '%Y -'
            label2 = ' %Y'
        elif rng >=3600*24*30*24:
            string = '%Y'
            label1 = ''
            label2 = ''
        for x in values:
            try:
                strns.append(time.strftime(string, time.localtime(x)))
            except ValueError:  ## Windows can't handle dates before 1970
                strns.append('')
        try:
            label = time.strftime(label1, time.localtime(min(values)))+time.strftime(label2, time.localtime(max(values)))
        except ValueError:
            label = ''
        #self.setLabel(text=label)
        return strns

class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)
        
    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            self.autoRange()
            
    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev)


app = pg.mkQApp()

axis = DateAxis(orientation='bottom')
vb = CustomViewBox()

pw = pg.PlotWidget(viewBox=vb, axisItems={'bottom': axis}, enableMenu=False, title="PlotItem with custom axis and ViewBox<br>Menu disabled, mouse behavior changed: left-drag to zoom, right-click to reset zoom")
dates = np.arange(8) * (3600*24*356)
pw.plot(x=dates, y=[1,6,2,4,3,5,6,8], symbol='o')
pw.show()
pw.setWindowTitle('pyqtgraph example: customPlot')

r = pg.PolyLineROI([(0,0), (10, 10)])
pw.addItem(r)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()







######################################### MULTIPLE PLOTS SPEED TEST #######################################

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

app = pg.mkQApp()
plt = pg.PlotWidget()

app.processEvents()

## Putting this at the beginning or end does not have much effect
plt.show()   

## The auto-range is recomputed after each item is added,
## so disabling it before plotting helps
plt.enableAutoRange(False, False)

def plot():
    start = pg.ptime.time()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts)*0.8
    for i in xrange(n):
        for j in xrange(n):
            ## calling PlotWidget.plot() generates a PlotDataItem, which 
            ## has a bit more overhead than PlotCurveItem, which is all 
            ## we need here. This overhead adds up quickly and makes a big
            ## difference in speed.
            
            #plt.plot(x=x+i, y=y+j)
            plt.addItem(pg.PlotCurveItem(x=x+i, y=y+j))
            
            #path = pg.arrayToQPath(x+i, y+j)
            #item = QtGui.QGraphicsPathItem(path)
            #item.setPen(pg.mkPen('w'))
            #plt.addItem(item)
            
    dt = pg.ptime.time() - start
    print("Create plots took: %0.3fms" % (dt*1000))

## Plot and clear 5 times, printing the time it took
for i in range(5):
    plt.clear()
    plot()
    app.processEvents()
    plt.autoRange()





def fastPlot():
    ## Different approach:  generate a single item with all data points.
    ## This runs about 20x faster.
    start = pg.ptime.time()
    n = 15
    pts = 100
    x = np.linspace(0, 0.8, pts)
    y = np.random.random(size=pts)*0.8
    xdata = np.empty((n, n, pts))
    xdata[:] = x.reshape(1,1,pts) + np.arange(n).reshape(n,1,1)
    ydata = np.empty((n, n, pts))
    ydata[:] = y.reshape(1,1,pts) + np.arange(n).reshape(1,n,1)
    conn = np.ones((n*n,pts))
    conn[:,-1] = False # make sure plots are disconnected
    path = pg.arrayToQPath(xdata.flatten(), ydata.flatten(), conn.flatten())
    item = QtGui.QGraphicsPathItem(path)
    item.setPen(pg.mkPen('w'))
    plt.addItem(item)
    
    dt = pg.ptime.time() - start
    print("Create plots took: %0.3fms" % (dt*1000))


## Plot and clear 5 times, printing the time it took
if hasattr(pg, 'arrayToQPath'):
    for i in range(5):
        plt.clear()
        fastPlot()
        app.processEvents()
else:
    print("Skipping fast tests--arrayToQPath function is missing.")

plt.autoRange()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        
        
        

######################################### LOG AXIS #######################################
#Test programmatically setting log transformation modes.
#"""
        
import initExample ## Add path to library (just for examples; you do not need this)

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg


app = QtGui.QApplication([])

w = pg.GraphicsWindow()
w.setWindowTitle('pyqtgraph example: logAxis')
p1 = w.addPlot(0,0, title="X Semilog")
p2 = w.addPlot(1,0, title="Y Semilog")
p3 = w.addPlot(2,0, title="XY Log")
p1.showGrid(True, True)
p2.showGrid(True, True)
p3.showGrid(True, True)
p1.setLogMode(True, False)
p2.setLogMode(False, True)
p3.setLogMode(True, True)
w.show()

y = np.random.normal(size=1000)
x = np.linspace(0, 1, 1000)
p1.plot(x, y)
p2.plot(x, y)
p3.plot(x, y)



#p.getAxis('bottom').setLogMode(True)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()




######################################### DATA SLICING #######################################
#Demonstrate a simple data-slicing task: given 3D data (displayed at top), select 
#a 2D plane and interpolate data along that plane to generate a slice image 
#(displayed at bottom). 
#"""

## Add path to library (just for examples; you do not need this)
import initExample

import numpy as np
import scipy
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with two ImageView widgets
win = QtGui.QMainWindow()
win.resize(800,800)
win.setWindowTitle('pyqtgraph example: DataSlicing')
cw = QtGui.QWidget()
win.setCentralWidget(cw)
l = QtGui.QGridLayout()
cw.setLayout(l)
imv1 = pg.ImageView()
imv2 = pg.ImageView()
l.addWidget(imv1, 0, 0)
l.addWidget(imv2, 1, 0)
win.show()

roi = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
imv1.addItem(roi)

x1 = np.linspace(-30, 10, 128)[:, np.newaxis, np.newaxis]
x2 = np.linspace(-20, 20, 128)[:, np.newaxis, np.newaxis]
y = np.linspace(-30, 10, 128)[np.newaxis, :, np.newaxis]
z = np.linspace(-20, 20, 128)[np.newaxis, np.newaxis, :]
d1 = np.sqrt(x1**2 + y**2 + z**2)
d2 = 2*np.sqrt(x1[::-1]**2 + y**2 + z**2)
d3 = 4*np.sqrt(x2**2 + y[:,::-1]**2 + z**2)
data = (np.sin(d1) / d1**2) + (np.sin(d2) / d2**2) + (np.sin(d3) / d3**2)

def update():
    global data, imv1, imv2
    d2 = roi.getArrayRegion(data, imv1.imageItem, axes=(1,2))
    imv2.setImage(d2)
    
roi.sigRegionChanged.connect(update)


## Display the data
imv1.setImage(data)
imv1.setHistogramRange(-0.01, 0.01)
imv1.setLevels(-0.003, 0.003)

update()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()








######################################### PLOT AUTORANGE #######################################
import initExample ## Add path to library (just for examples; you do not need this)


from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="Plot auto-range examples")
win.resize(800,600)
win.setWindowTitle('pyqtgraph example: PlotAutoRange')

d = np.random.normal(size=100)
d[50:54] += 10
p1 = win.addPlot(title="95th percentile range", y=d)
p1.enableAutoRange('y', 0.95)


p2 = win.addPlot(title="Auto Pan Only")
p2.setAutoPan(y=True)
curve = p2.plot()
def update():
    t = pg.time()
    
    data = np.ones(100) * np.sin(t)
    data[50:60] += np.sin(t)
    global curve
    curve.setData(data)
    
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()







######################################### PLOT AUTORANGE #######################################
import initExample ## Add path to library (just for examples; you do not need this)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np


## Create image to display
arr = np.ones((100, 100), dtype=float)
arr[45:55, 45:55] = 0
arr[25, :] = 5
arr[:, 25] = 5
arr[75, :] = 5
arr[:, 75] = 5
arr[50, :] = 10
arr[:, 50] = 10
arr += np.sin(np.linspace(0, 20, 100)).reshape(1, 100)
arr += np.random.normal(size=(100,100))


## create GUI
app = QtGui.QApplication([])
w = pg.GraphicsWindow(size=(800,800), border=True)
w.setWindowTitle('pyqtgraph example: ROI Examples')

text = """Data Selection From Image.<br>\n
Drag an ROI or its handles to update the selected image.<br>
Hold CTRL while dragging to snap to pixel boundaries<br>
and 15-degree rotation angles.
"""
w1 = w.addLayout(row=0, col=0)
label1 = w1.addLabel(text, row=0, col=0)
v1a = w1.addViewBox(row=1, col=0, lockAspect=True)
v1b = w1.addViewBox(row=2, col=0, lockAspect=True)
img1a = pg.ImageItem(arr)
v1a.addItem(img1a)
img1b = pg.ImageItem()
v1b.addItem(img1b)
v1a.disableAutoRange('xy')
v1b.disableAutoRange('xy')
v1a.autoRange()
v1b.autoRange()

rois = []
rois.append(pg.RectROI([20, 20], [20, 20], pen=(0,9)))
rois[-1].addRotateHandle([1,0], [0.5, 0.5])
rois.append(pg.LineROI([0, 60], [20, 80], width=5, pen=(1,9)))
rois.append(pg.MultiRectROI([[20, 90], [50, 60], [60, 90]], width=5, pen=(2,9)))
rois.append(pg.EllipseROI([60, 10], [30, 20], pen=(3,9)))
rois.append(pg.CircleROI([80, 50], [20, 20], pen=(4,9)))
#rois.append(pg.LineSegmentROI([[110, 50], [20, 20]], pen=(5,9)))
#rois.append(pg.PolyLineROI([[110, 60], [20, 30], [50, 10]], pen=(6,9)))

def update(roi):
    img1b.setImage(roi.getArrayRegion(arr, img1a), levels=(0, arr.max()))
    v1b.autoRange()
    
for roi in rois:
    roi.sigRegionChanged.connect(update)
    v1a.addItem(roi)

update(rois[-1])
    


text = """User-Modifiable ROIs<br>
Click on a line segment to add a new handle.
Right click on a handle to remove.
"""
w2 = w.addLayout(row=0, col=1)
label2 = w2.addLabel(text, row=0, col=0)
v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
v2a.addItem(r2a)
r2b = pg.PolyLineROI([[0,-20], [10,-10], [10,-30]], closed=False)
v2a.addItem(r2b)
v2a.disableAutoRange('xy')
#v2b.disableAutoRange('xy')
v2a.autoRange()
#v2b.autoRange()

text = """Building custom ROI types<Br>
ROIs can be built with a variety of different handle types<br>
that scale and rotate the roi around an arbitrary center location
"""
w3 = w.addLayout(row=1, col=0)
label3 = w3.addLabel(text, row=0, col=0)
v3 = w3.addViewBox(row=1, col=0, lockAspect=True)

r3a = pg.ROI([0,0], [10,10])
v3.addItem(r3a)
## handles scaling horizontally around center
r3a.addScaleHandle([1, 0.5], [0.5, 0.5])
r3a.addScaleHandle([0, 0.5], [0.5, 0.5])

## handles scaling vertically from opposite edge
r3a.addScaleHandle([0.5, 0], [0.5, 1])
r3a.addScaleHandle([0.5, 1], [0.5, 0])

## handles scaling both vertically and horizontally
r3a.addScaleHandle([1, 1], [0, 0])
r3a.addScaleHandle([0, 0], [1, 1])

r3b = pg.ROI([20,0], [10,10])
v3.addItem(r3b)
## handles rotating around center
r3b.addRotateHandle([1, 1], [0.5, 0.5])
r3b.addRotateHandle([0, 0], [0.5, 0.5])

## handles rotating around opposite corner
r3b.addRotateHandle([1, 0], [0, 1])
r3b.addRotateHandle([0, 1], [1, 0])

## handles rotating/scaling around center
r3b.addScaleRotateHandle([0, 0.5], [0.5, 0.5])
r3b.addScaleRotateHandle([1, 0.5], [0.5, 0.5])

v3.disableAutoRange('xy')
v3.autoRange()


text = """Transforming objects with ROI"""
w4 = w.addLayout(row=1, col=1)
label4 = w4.addLabel(text, row=0, col=0)
v4 = w4.addViewBox(row=1, col=0, lockAspect=True)
g = pg.GridItem()
v4.addItem(g)
r4 = pg.ROI([0,0], [100,100])
r4.addRotateHandle([1,0], [0.5, 0.5])
r4.addRotateHandle([0,1], [0.5, 0.5])
img4 = pg.ImageItem(arr)
v4.addItem(r4)
img4.setParentItem(r4)

v4.disableAutoRange('xy')
v4.autoRange()








## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()















