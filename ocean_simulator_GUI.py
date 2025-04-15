import time
import argparse
import glob

import numpy as np
np.bool = np.bool_

from scipy import ndimage
from PyQt5 import QtWidgets, QtCore

from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
import h5py
from scipy.fft import fft, ifft,fftfreq
import scipy.interpolate
from scipy.ndimage import gaussian_filter
from multiprocessing.pool import ThreadPool

# Define the parser
parser = argparse.ArgumentParser(description='Mixing Simulator')
parser.add_argument('-n', action="store", dest='Np', default=2**8)
parser.add_argument('-p', action="store", dest='p', default=3)
args = parser.parse_args()

Np=int(args.Np)
p=int(args.p)

tmax=2000
t0=-24*31*4
t0=-24*10*4
t0=0

origin=[150,150]
origin=[0,50]
n=330

version=' v1.0 08/2024'


IMAGE_SHAPE = (n,n)  # (height, width)
CANVAS_SIZE = (n*9,n*3)  # (width, height)

COLORMAP_CHOICES = ["viridis", 'binary', 'gist_gray', 'plasma', 'inferno', 'magma', 'cividis',"reds", "blues"]
SIMULATION_CHOICES = ["Hachures","Vide"]
IMAGE_CHOICES = ["Scalar","Vitesse","Temperature"]

FOLDER='./'

class Controls(QtWidgets.QWidget):
	def __init__(self, parent=None):
		super().__init__(parent)
		
		# Sine flow parameters
		self.power = -5/3.
		self.lmin = 0.01
		self.lmax= 1.0
		self.a = 0.75
		self.D=-9
		self.mode='Source'
		self.speed=0.05
		self.s=int(Np*0.05)
		self.tcorr=20
		self.fps=1
		self.mode="Gradient"
		self.start=True
		self.quit=False
		self.imtype="Scalar"
		
		layout = QtWidgets.QVBoxLayout()
		
		self.title =  QtWidgets.QLabel("\n  Ocean Simulator \n "+version +"\n"+"J. Heyman \n données MARS3D/ifremer \n")
		layout.addWidget(self.title)
		
		
		self.start_bt = QtWidgets.QPushButton('Start/Pause', self)
		layout.addWidget(self.start_bt)
		
# 		self.quit_bt = QtWidgets.QPushButton('Quit', self)
# 		layout.addWidget(self.quit_bt)
		
# 		self.pause_chooser = QtWidgets.QComboBox()
# 		self.pause_chooser.addItems(SIMULATION_CHOICES)
# 		layout.addWidget(self.pause_chooser)
		
		self.mode_label = QtWidgets.QLabel("Condition initiale:")
		layout.addWidget(self.mode_label)
		self.mode_chooser = QtWidgets.QComboBox()
		self.mode_chooser.addItems(SIMULATION_CHOICES)
		layout.addWidget(self.mode_chooser)
		
		
		self.imtype_label = QtWidgets.QLabel("Visualisation:")
		layout.addWidget(self.imtype_label)
		self.imtype_chooser = QtWidgets.QComboBox()
		self.imtype_chooser.addItems(IMAGE_CHOICES)
		layout.addWidget(self.imtype_chooser)
		
		self.colormap_label = QtWidgets.QLabel("Colormap:")
		layout.addWidget(self.colormap_label)
		self.colormap_chooser = QtWidgets.QComboBox()
		self.colormap_chooser.addItems(COLORMAP_CHOICES)
		layout.addWidget(self.colormap_chooser)
		
		self.rescale_bt = QtWidgets.QPushButton('Rescale', self)
		layout.addWidget(self.rescale_bt)
		# Slider
		#layout2=QtWidgets.QHBoxLayout()
		# power (x100)
		
		self.returne =  QtWidgets.QLabel("\n")
		layout.addWidget(self.returne)
		
		self.time_label = QtWidgets.QLabel("Date :")
		self.time_label2 = QtWidgets.QLabel("")
		layout.addWidget(self.time_label)
		layout.addWidget(self.time_label2)
		
		self.returne =  QtWidgets.QLabel("\n")
		layout.addWidget(self.returne)
		
		self.coeff_label = QtWidgets.QLabel("Coeff Marée :")
		layout.addWidget(self.coeff_label)
		
		self.returne =  QtWidgets.QLabel("\n")
		layout.addWidget(self.returne)
		# Speed slider
		self.fps_label = QtWidgets.QLabel("Jours / secondes: {:1.1f}".format(self.fps))
		layout.addWidget(self.fps_label)
		self.fps_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.fps_sl.setMinimum(1)
		self.fps_sl.setMaximum(30)
		self.fps_sl.setValue(10)
		self.fps_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.fps_sl.setTickInterval(5)
		layout.addWidget(self.fps_sl)
		
		# Roughness slider
# 		self.power_label = QtWidgets.QLabel("Rougness: {:1.2f}".format(self.power))
# 		layout.addWidget(self.power_label)
# 		self.power_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# 		self.power_sl.setMinimum(-500)
# 		self.power_sl.setMaximum(0)
# 		self.power_sl.setValue(-83)
# 		self.power_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
# 		self.power_sl.setTickInterval(100)
# 		layout.addWidget(self.power_sl)
		
		
		self.D_label = QtWidgets.QLabel("Diffusion (log): {:1.1f}".format(self.D))
		layout.addWidget(self.D_label)
		self.D_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
		self.D_sl.setMinimum(-90)
		self.D_sl.setMaximum(50)
		self.D_sl.setValue(-90)
		self.D_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
		self.D_sl.setTickInterval(10)
		layout.addWidget(self.D_sl)

		# Amplitude slider
# 		self.a_label = QtWidgets.QLabel("Amplitude: {:1.2f}".format(self.a))
# 		layout.addWidget(self.a_label)
# 		self.a_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# 		self.a_sl.setMinimum(0)
# 		self.a_sl.setMaximum(50)
# 		self.a_sl.setValue(5)
# 		self.a_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
# 		self.a_sl.setTickInterval(10)
# 		layout.addWidget(self.a_sl)
		
		# Correlation time
# 		self.tcorr_label = QtWidgets.QLabel("Corr. time: {:1.2f}".format(self.tcorr))
# 		layout.addWidget(self.tcorr_label)
# 		self.tcorr_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# 		self.tcorr_sl.setMinimum(0)
# 		self.tcorr_sl.setMaximum(50)
# 		self.tcorr_sl.setValue(5)
# 		self.tcorr_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
# 		self.tcorr_sl.setTickInterval(10)
# 		layout.addWidget(self.tcorr_sl)
# 		
# 		# Max lengthscale
# 		self.lmax_label = QtWidgets.QLabel("Max Lengthscales: {:1.2f}".format(self.lmax))
# 		layout.addWidget(self.lmax_label)
# 		self.lmax_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# 		self.lmax_sl.setMinimum(0)
# 		self.lmax_sl.setMaximum(100)
# 		self.lmax_sl.setValue(100)
# 		self.lmax_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
# 		self.lmax_sl.setTickInterval(20)
# 		layout.addWidget(self.lmax_sl)

# 		# Min lengthscale
# 		self.lmin_label = QtWidgets.QLabel("Min Lengthscales: {:1.2f}".format(self.lmin))
# 		layout.addWidget(self.lmin_label)
# 		self.lmin_sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
# 		self.lmin_sl.setMinimum(1)
# 		self.lmin_sl.setMaximum(100)
# 		self.lmin_sl.setValue(1)
# 		self.lmin_sl.setTickPosition(QtWidgets.QSlider.TicksAbove)
# 		self.lmin_sl.setTickInterval(20)
# 		layout.addWidget(self.lmin_sl)

		#layout.SetMaximumSize
		
		layout.addStretch(1)
		self.setLayout(layout)

	def set_amplitude(self,r):
		self.a=r/10
		self.a_label.setText("Amplitude: {:1.1f}".format(self.a))

	def set_diffusion(self,r):
		self.D=r/10
		self.D_label.setText("Diffusion (log): {:1.2f}".format(self.D))
	
	def set_roughness(self,r):
		self.power=r/100
		self.power_label.setText("Rougness: {:1.2f}".format(self.power))
		
	def set_tcorr(self,r):
		self.tcorr=r
		self.tcorr_label.setText("Corr time: {:1.2f}".format(self.tcorr))
		
	def set_lmax(self,r):
		self.lmax=r/100
		self.lmax_label.setText("Max Lengthscale: {:1.2f}".format(self.lmax))
		
	def set_lmin(self,r):
		self.lmin=r/100
		self.lmin_label.setText("Min Lengthscale: {:1.2f}".format(self.lmin))
		
	def set_fps(self,r):
		self.fps=r/10
		self.fps_label.setText("Jours / seconde: {:1.1f}".format(self.fps))

	def set_mode(self, _mode: str):
		self.mode = _mode

	def set_start(self):
		self.start=not(self.start)

	def set_quit(self):
		self.quit=True
		
	def set_imtype(self, imt: str):
		self.imtype = imt
		
class CanvasWrapper:
	def __init__(self):
		
		self.canvas = SceneCanvas(size=CANVAS_SIZE)
#		self.grid = self.canvas.central_widget.add_grid()
	
		self.view_top = self.canvas.central_widget.add_view()

#		self.view_top = self.grid.add_view(0, 0, bgcolor='cyan')
		image_data = np.zeros(IMAGE_SHAPE)
		
		
		self.image = visuals.Image(
			image_data,
			texture_format="auto",
			clim=[-1,1],
			cmap=COLORMAP_CHOICES[0],
			parent=self.view_top.scene,
			interpolation='bilinear'
		)
		
		#self.view_top.camera.PanZoomCamera(parent=self.view_top.scene, aspect=1, name='panzoom')
		self.view_top.camera = "panzoom"
		#self.view_top.camera = cameras.base_camera.BaseCamera(aspect=1,interactive=False)
		self.view_top.camera.set_range(x=(0, IMAGE_SHAPE[1]), y=(0, IMAGE_SHAPE[0]), margin=0)
		self.view_top.camera.interactive=False
		
		# Point source on mouse click
		self.blob=[]
		self.canvas.events.mouse_release.connect(self.set_blob)


# 		self.view_bot = self.grid.add_view(1, 0, bgcolor='#c0c0c0')
# 		line_data = _generate_random_line_positions(NUM_LINE_POINTS)
# 		self.line = visuals.Line(line_data, parent=self.view_bot.scene, color=LINE_COLOR_CHOICES[0])
# 		self.view_bot.camera = "panzoom"
# 		self.view_bot.camera.set_range(x=(0, NUM_LINE_POINTS), y=(0, 1))

	def set_blob(self,event):
		if event.button == 1:
			# left click
			transform = self.image.transforms.get_transform(map_to="canvas")
			img_x, img_y = transform.imap(event.pos)[:2]
			# optionally do the below to tell other handlers not to look at this event:
			#event.handled = True
			self.blob=[img_x,img_y,1]
		if event.button == 2:
			# left click
			transform = self.image.transforms.get_transform(map_to="canvas")
			img_x, img_y = transform.imap(event.pos)[:2]
			# optionally do the below to tell other handlers not to look at this event:
			#event.handled = True
			self.blob=[img_x,img_y,-1]

	
	def set_image_colormap(self, cmap_name: str):
		print(f"Changing image colormap to {cmap_name}")
		self.image.cmap = cmap_name

		
# 	def set_line_color(self, color):
# 		print(f"Changing line color to {color}")
# 		self.line.set_data(color=color)

	def update_data(self, new_data_dict):
		#print("Updating data...")
#		self.line.set_data(new_data_dict["line"])
		self.image.set_data(new_data_dict["image"])
		self.canvas.update()

class MyMainWindow(QtWidgets.QMainWindow):
	closing = QtCore.pyqtSignal()

	def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
		super().__init__(*args, **kwargs)

		central_widget = QtWidgets.QWidget()
		main_layout = QtWidgets.QHBoxLayout()

		self._controls = Controls()
		main_layout.addWidget(self._controls)
		self._canvas_wrapper = canvas_wrapper
		main_layout.addWidget(self._canvas_wrapper.canvas.native)

		central_widget.setLayout(main_layout)
		self.setCentralWidget(central_widget)

		self._connect_controls()

	def _connect_controls(self):
		self._controls.mode_chooser.currentTextChanged.connect(self._controls.set_mode)
		self._controls.colormap_chooser.currentTextChanged.connect(self._canvas_wrapper.set_image_colormap)
		self._controls.imtype_chooser.currentTextChanged.connect(self._controls.set_imtype)
#		self._controls.line_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_line_color)
#		self._controls.power_sl.valueChanged.connect(self._controls.set_roughness)
		self._controls.D_sl.valueChanged.connect(self._controls.set_diffusion)
# 		self._controls.a_sl.valueChanged.connect(self._controls.set_amplitude)
# 		self._controls.tcorr_sl.valueChanged.connect(self._controls.set_tcorr)
# 		self._controls.lmin_sl.valueChanged.connect(self._controls.set_lmin)
# 		self._controls.lmax_sl.valueChanged.connect(self._controls.set_lmax)
		self._controls.fps_sl.valueChanged.connect(self._controls.set_fps)
		self._controls.start_bt.clicked.connect(self._controls.set_start)
		#self._controls.quit_bt.clicked.connect(self._controls.set_quit)
		self._controls.rescale_bt.clicked.connect(self.set_rescale)

	def set_rescale(self):
		C=np.array(self._canvas_wrapper.image._data)
		self._canvas_wrapper.image.clim=[np.min(C),np.max(C)]
	
	def closeEvent(self, event):
		print("Closing main window!")
		self.closing.emit()
		return super().closeEvent(event)


class DataSource(QtCore.QObject):
	"""Object representing a complex data producer."""
	new_data = QtCore.pyqtSignal(dict)
	finished = QtCore.pyqtSignal()

	def __init__(self, myMainWindow: MyMainWindow, parent=None):
		super().__init__(parent)
		self._should_end = False
		self._image_data = np.zeros(IMAGE_SHAPE)
#		self._line_data = _generate_random_line_positions(NUM_LINE_POINTS)
		self._myMainWindow = myMainWindow
		self._tmax=1000
		self._D=10**self._myMainWindow._controls.D
#		self._power=self._myMainWindow._controls.power
# 		self._lmin=self._myMainWindow._controls.lmin
# 		self._lmax=self._myMainWindow._controls.lmax
# 		self._a=self._myMainWindow._controls.a
		self._mode=self._myMainWindow._controls.mode
#		self._s=self._myMainWindow._controls.s
#		self._tcorr=self._myMainWindow._controls.tcorr
		self._fps=self._myMainWindow._controls.fps
		self._imtype=self._myMainWindow._controls.imtype
		self._quit=self._myMainWindow._controls.quit
		self._flow='1D'
	
	def load_velocity(self,filename):

		File=h5py.File(filename,'r')
		#File=h5py.File('/data/Simulations/ifremer/All.h5','r')
		
		scale_factor=6.103702e-5
		#scale_factor=6.103702e-6
		
		#scale_factor=1.
		
		
		
		print('read velocities...')
		U=File['U'][t0:,origin[0]:origin[0]+n,origin[1]:origin[1]+n]/scale_factor
		V=File['V'][t0:,origin[0]:origin[0]+n,origin[1]:origin[1]+n]/scale_factor
		Temp=File['Temp'][t0:,origin[0]:origin[0]+n,origin[1]:origin[1]+n]
		Time=File['Time'][t0:]
		
		print('Done !')
		
		

		mask=np.isnan(U[0,:,:])
		
		U[np.isnan(U)]=0
		Temp[np.isnan(Temp)]=0
		V[np.isnan(V)]=0
		
		
		T=np.arange(0,U.shape[0])*3600
		
		#linear interpolant
# 		print('build interpolator...')
# 		uf=scipy.interpolate.interp1d(T,U,axis=0,kind='nearest')
# 		vf=scipy.interpolate.interp1d(T,V,axis=0,kind='nearest')
# 		tf=scipy.interpolate.interp1d(T,Temp,axis=0,kind='nearest')
# 		return uf,vf,tf,Time,T
		return U,V,Temp,Time,T,mask
	
	def run_data_creation(self):
		# local parameters
		
		#print(Np*p,self._tmax,self._tcorr)
		# Load velocity
				
		flist=glob.glob(FOLDER+"/*.h5")
		print(flist)
		if len(flist)==12:
			ALLMONTHS=True
		else:
			ALLMONTHS=False

		month=0
		filename=FOLDER+'All_2023_{:02d}.h5'.format(month+1)
		#uf,vf,tf,Time,T=self.load_velocity(filename)
# 		u=uf(0)
# 		v=vf(0)
# 		mask=np.isnan(u)
		U,V,Temp,Time,T,mask=self.load_velocity(filename)
		u=U[0,:,:]
		v=V[0,:,:]
		
		# load next month in a separate thread
		if ALLMONTHS:
			month2=month+1
			filename2=FOLDER+'All_2023_{:02d}.h5'.format(month2+1)
			pool = ThreadPool(processes=1)
			thread = pool.apply_async(self.load_velocity, (filename2,))
		#thread.get()
		else:
			print('Running only for first month.')
		
		
		import pandas as pd
		try:
			D=pd.read_csv(FOLDER+'coefficients_2023.csv')
			date=Time[0].decode('UTF-8')[:10]
			coeff=D['coeff'].get(D['Date']==date).to_numpy()[0]
		except:
			coeff=-1
		
		method='laxW'
		method='1st'
		method='fluxlim'
		limiter='superbee'
		#limiter='minmod'
		
		
		dx=2500 #m
		
		x,y=np.meshgrid(np.arange(n)*dx,np.arange(n)*dx)
		
		#frequency domain for diffusion
		ky=2*np.pi*np.tile(fftfreq(n, d=1.0/n),(n,1)).T
		kx=2*np.pi*np.tile(fftfreq(n, d=1.0/n),(n,1))
		k=np.sqrt(ky**2+kx**2)
		
		
		
		
		nbnd=3
		
		# Boundary field
		isbnd=np.zeros((n+2*nbnd,n+2*nbnd),dtype=bool)
		
		isbnd[nbnd:-nbnd,nbnd:-nbnd]=mask
		# side
		isbnd[:nbnd,:]=True
		isbnd[-nbnd:,:]=True
		isbnd[:,:nbnd]=True
		isbnd[:,-nbnd:]=True
		
		 # island
		#isbnd[nbnd:-nbnd,nbnd:-nbnd][(x-n/3*dx)**2+(y-n/2*dx)**2<0.1]=True
		isfluid=~isbnd
		# perform distance transform to fill boundary with local values
		BND_dist,BND_ind=ndimage.distance_transform_edt(isbnd, return_indices=True)
		
		isbnd_plot=np.float32(np.copy(isbnd))
		isbnd_plot[~isbnd]=np.nan
		
		def add_bnd_per(cbnd,c,n,axis=np.array([0,1])):
			if np.any(axis==1):
				cbnd[n:-n,:n]=c[:,-n:]
				cbnd[n:-n,-n:]=c[:,:n]
			if np.any(axis==0):
				cbnd[:n,n:-n]=c[-n:,:]
				cbnd[-n:,n:-n]=c[:n,:]
			return cbnd
		
		
		def add_bnd_noflux(cbnd,c,n,axis=np.array([0,1])):
			if np.any(axis==1):
				cbnd[n:-n,:n]=np.tile(c[:,n].reshape(-1,1),(1,n))
				cbnd[n:-n,-n:]=np.tile(c[:,-n].reshape(-1,1),(1,n))
			if np.any(axis==0):
				cbnd[:n,n:-n]=np.tile(c[n,:].reshape(1,-1),(n,1))
				cbnd[-n:,n:-n]=np.tile(c[-n,:].reshape(1,-1),(n,1))
			return cbnd
		
		def u_per(u,mode='fwd'):
			up=np.zeros(len(u)+1)
			up[1:]=u
			up[0]=u[-1]
			if mode=='fwd':
				return up[1:]
			if mode=='bwd':
				return up[0:-1]
		
		def minmod(a,b):
			mm=np.copy(a)
			cond1=(np.abs(a)<=np.abs(b))*((a*b)>0)
			mm[cond1]=a[cond1]
			cond1=(np.abs(a)>np.abs(b))*((a*b)>0)
			mm[cond1]=b[cond1]
			cond1=(a*b)<0
			mm[cond1]=0
			return mm
		
		
		
		np.random.seed(1)
		x,y=np.meshgrid(np.arange(n)*dx,np.arange(n)*dx)
		
		#frequency domain for diffusion
		ky=2*np.pi*np.tile(fftfreq(n+2*nbnd, d=dx),(n+2*nbnd,1)).T
		kx=2*np.pi*np.tile(fftfreq(n+2*nbnd, d=dx),(n+2*nbnd,1))
		k=np.sqrt(ky**2+kx**2)
		
		
		
		cfl=0.99
		dt=cfl/np.max(np.sqrt(u**2+v**2))*dx
		
		print('dt=',dt)
		
		c0=np.zeros((n,n))
		for kc in np.linspace(0,1,10):
			c0+=np.exp(-((y-n*dx*kc)**2)/(dx**2*100))
		#c0=2*c0/c0.max()-1
		#c0=np.zeros((n,n))
		#c0[(x-n/3*dx)**2+(y-n/2*dx)**2<0.1]=1
		#v=np.sin(x*n*dx/2/np.pi)
		#c0=np.zeros((n,n))
		#c0[y>n*dx/2]=1
		
		# Start sim
		#c1=np.copy(c0)
		ones=np.ones(c0.shape)
		
		t=0
		treel=0
		#tmax=t+T.shape[0]*3600
		
		it=0
		m=[]
		c0bnd=np.zeros((c0.shape[0]+2*nbnd,c0.shape[1]+2*nbnd))
		rescale=False
		
		while True:
			
			# local parameters
			self._D=10**self._myMainWindow._controls.D
			self._power=self._myMainWindow._controls.power
			self._lmax=self._myMainWindow._controls.lmax
			
			if self._myMainWindow._controls.lmin>=self._lmax*0.9:
				self._myMainWindow._controls.lmin_sl.setValue(1)
				self._lmin=0.01
			else:				
				self._lmin=self._myMainWindow._controls.lmin
				
			self._a=self._myMainWindow._controls.a
			if self._mode!=self._myMainWindow._controls.mode:
				rescale=True
				it0=it
				if self._myMainWindow._controls.mode==SIMULATION_CHOICES[1]:
					c0=np.zeros((n,n))
				if self._myMainWindow._controls.mode==SIMULATION_CHOICES[0]:
					c0=np.zeros((n,n))
					for kc in np.linspace(0,1,10):
						c0+=np.exp(-((y-n*dx*kc)**2)/(dx**2*100))
					#c0=2*c0/c0.max()-1

				self._mode=self._myMainWindow._controls.mode
				
			self._s=self._myMainWindow._controls.s
			self._fps=self._myMainWindow._controls.fps
			
			
			if self._imtype!=self._myMainWindow._controls.imtype:
				self._imtype=self._myMainWindow._controls.imtype
				rescale=True
				it0=it
			
			self._quit=self._myMainWindow._controls.quit

			diffusion=self._D
			while not(self._myMainWindow._controls.start):
				time.sleep(1)
				
			tini=time.time()
			
			if treel>(len(T)-1)*3600:
				if ALLMONTHS:
					treel=treel-(len(T)-1)*3600
					U,V,Temp,Time,T,mask = thread.get() # get new month from thread
					# load new month from separate thread 
					month=np.mod(month+1,12)
					month2=np.mod(month+1,12)
					filename2=FOLDER+'All_2023_{:02d}.h5'.format(month2+1)
					#pool = ThreadPool(processes=1)
					thread = pool.apply_async(self.load_velocity, (filename2,))
					#uf,vf,tf,Time,T=self.load_velocity(filename)
					#U,V,Temp,Time,T,mask=self.load_velocity(filename)
				else:
					treel=0

#			treel=np.mod(t,(len(T)-1)*3600)
# 			u=uf(treel)
# 			v=vf(treel)
		
					
			u=U[int(treel/3600),:,:]
			v=V[int(treel/3600),:,:]
			cfl=0.99
			dt=cfl/np.max(np.sqrt(u**2+v**2))*dx
			
			self._myMainWindow._controls.time_label2.setText(Time[int(treel/3600)].decode('UTF-8')[:19])
			
			date=Time[int(treel/3600)].decode('UTF-8')[:10]
			try:
				coeff=D['coeff'].get(D['Date']==date).to_numpy()[0]
			except:
				coeff=-1
				
			self._myMainWindow._controls.coeff_label.setText('Coeff Marée : {:d}'.format(coeff))
			
			#print(u.max(),v.max())
			# Boundary conditions
			#
			c0bnd[nbnd:-nbnd,nbnd:-nbnd]=c0
			c0bnd=c0bnd[BND_ind[0],BND_ind[1]] # no flux
			#cbnd=add_bnd_per(c0bnd,c0,nbnd,axis=np.array([0,1]))
			
			if method=='fluxlim':
				
				# x
				#vel=u
				
				dQ_m3_2=np.diff(c0bnd,axis=1)[nbnd:-nbnd,nbnd-2:-nbnd-1]
				dQ_p1_2=np.diff(c0bnd,axis=1)[nbnd:-nbnd,nbnd:-nbnd+1]
				dQ_m1_2=np.diff(c0bnd,axis=1)[nbnd:-nbnd,nbnd-1:-nbnd]
				dQ_p3_2=np.diff(c0bnd,axis=1)[nbnd:-nbnd,nbnd+1:-nbnd+2]
				
				# theta_i-1/2
				theta_m1_2=dQ_m3_2/dQ_m1_2
				theta_m1_2[u<0]=dQ_p1_2[u<0]/dQ_m1_2[u<0]
				
				# theta_i+1/2
				theta_p1_2=dQ_m1_2/dQ_p1_2
				theta_p1_2[u<0]=dQ_p3_2[u<0]/dQ_p1_2[u<0]
				
				theta_m1_2[~np.isfinite(theta_m1_2)]=1.
				theta_p1_2[~np.isfinite(theta_p1_2)]=1.
				
				#Minmod
				if limiter=='minmod':
					phi_m1_2=minmod(ones,theta_m1_2) 
					phi_p1_2=minmod(ones,theta_p1_2)
				
				if limiter=='superbee':
					phi_m1_2=np.maximum(np.maximum(0,np.minimum(1,2*theta_m1_2)),np.minimum(2,theta_m1_2))
					phi_p1_2=np.maximum(np.maximum(0,np.minimum(1,2*theta_p1_2)),np.minimum(2,theta_p1_2))
		
				nu=u*dt/dx
				
				dc=-nu*dQ_m1_2-1/2.*nu*(1-nu)*(phi_p1_2*dQ_p1_2-phi_m1_2*dQ_m1_2)
				dc_neg=-nu*dQ_p1_2+1/2.*nu*(1+nu)*(phi_p1_2*dQ_p1_2-phi_m1_2*dQ_m1_2)
				dc[u<0]=dc_neg[u<0]
				c0=c0+dc
				
				# y
				# Boundary conditions
				#c0bnd=np.zeros((c0.shape[0]+2*nbnd,c0.shape[1]+2*nbnd))
				c0bnd[nbnd:-nbnd,nbnd:-nbnd]=c0
				c0bnd=c0bnd[BND_ind[0],BND_ind[1]] # no flux
				#cbnd=add_bnd_noflux(c0bnd,c0,nbnd,axis=np.array([0,1]))
				
				#vel=v
				
				dQ_m3_2=np.diff(c0bnd,axis=0)[nbnd-2:-nbnd-1,nbnd:-nbnd]
				dQ_p1_2=np.diff(c0bnd,axis=0)[nbnd:-nbnd+1,nbnd:-nbnd]
				dQ_m1_2=np.diff(c0bnd,axis=0)[nbnd-1:-nbnd,nbnd:-nbnd]
				dQ_p3_2=np.diff(c0bnd,axis=0)[nbnd+1:-nbnd+2,nbnd:-nbnd]
				
				# theta_i-1/2
				theta_m1_2=dQ_m3_2/dQ_m1_2
				theta_m1_2[v<0]=dQ_p1_2[v<0]/dQ_m1_2[v<0]
				
				# theta_i+1/2
				theta_p1_2=dQ_m1_2/dQ_p1_2
				theta_p1_2[v<0]=dQ_p3_2[v<0]/dQ_p1_2[v<0]
				
				theta_m1_2[~np.isfinite(theta_m1_2)]=1.
				theta_p1_2[~np.isfinite(theta_p1_2)]=1.
				
				#Minmod
				if limiter=='minmod':
					phi_m1_2=minmod(ones,theta_m1_2) 
					phi_p1_2=minmod(ones,theta_p1_2)
				
				if limiter=='superbee':
					phi_m1_2=np.maximum(np.maximum(0,np.minimum(1,2*theta_m1_2)),np.minimum(2,theta_m1_2))
					phi_p1_2=np.maximum(np.maximum(0,np.minimum(1,2*theta_p1_2)),np.minimum(2,theta_p1_2))
		
				nu=v*dt/dx
				dc=-nu*dQ_m1_2-1/2.*nu*(1-nu)*(phi_p1_2*dQ_p1_2-phi_m1_2*dQ_m1_2)
				dc_neg=-nu*dQ_p1_2+1/2.*nu*(1+nu)*(phi_p1_2*dQ_p1_2-phi_m1_2*dQ_m1_2)
				dc[v<0]=dc_neg[v<0]
				c0=c0+dc
		
			# Spectral
			if diffusion>=1:
				
			# Boundary conditions
				c0bnd[nbnd:-nbnd,nbnd:-nbnd]=c0
				c0bnd=c0bnd[BND_ind[0],BND_ind[1]] # no flux
				c0bnd=add_bnd_per(c0bnd,c0,nbnd,axis=np.array([0,1]))
				
				c0f=fft(fft(c0bnd,axis=0),axis=1)
				c0f=c0f*np.exp(-k**2*diffusion*dt)
				c0bnd=np.real(ifft(ifft(c0f,axis=1),axis=0))
				
				c0=c0bnd[nbnd:-nbnd,nbnd:-nbnd]
				#print(c0.shape)
			# Non periodic diffusion
			# Explicit
			#c0=c0+dt/dx**2*diffusion*(c0bnd[nbnd-1:-nbnd-1,nbnd:-nbnd]-2*c0bnd[nbnd:-nbnd,nbnd:-nbnd]+c0bnd[nbnd+1:-nbnd+1,nbnd:-nbnd] +
			#								c0bnd[nbnd:-nbnd,nbnd-1:-nbnd-1]-2*c0bnd[nbnd:-nbnd,nbnd:-nbnd]+c0bnd[nbnd:-nbnd,nbnd+1:-nbnd+1])
		
		
			t=t+dt
			treel=treel+dt
			it=it+1
		#Flux limiters
		# 	W_m1_2= amdq  # Limited version of the wave
		# 	W_p1_2= amdq # Limited version of the wave
		# 	fi_m1_2=1/2*(np.abs(wave_speed_bw)*(1-dt/dx*np.abs(wave_speed_bw))*W_m1_2)
		# 	fi_p1_2
			print(it,' - t (hrs)=',t/3600.,' - Mass =',np.sum(c0)*dx**2,'time',time.time()-tini)
			m.append(np.sum(c0)*dx**2)
			
			#print(self._imtype)
			if self._imtype=="Scalar":
				
				#print(c0.max(),c0.min())
				cplot=c0*isfluid[nbnd:-nbnd,nbnd:-nbnd]
				cplot[isbnd[nbnd:-nbnd,nbnd:-nbnd]]=-1
				#print(cplot.shape)
				data_dict = { "image": cplot }
			if self._imtype=="Vitesse":
				vmag = np.sqrt(u**2+v**2)
				#print(vmag.shape)
				data_dict = { "image": vmag}
			if self._imtype=="Temperature":
				temp=(Temp[int(treel/3600),:,:]+c0*5)*isfluid[nbnd:-nbnd,nbnd:-nbnd]
				temp[temp<9]=9
				#print(temp.min(),temp.max())
				#print(temp.shape)
				data_dict = { "image": temp}
			
			self.new_data.emit(data_dict)
			
			if rescale:
				self._myMainWindow.set_rescale()
				print(rescale)
				if it>it0+10:
					rescale=False
				
			if len(self._myMainWindow._canvas_wrapper.blob)>0:
				s=5
				x=int(self._myMainWindow._canvas_wrapper.blob[0])
				y=int(self._myMainWindow._canvas_wrapper.blob[1])
				c00=self._myMainWindow._canvas_wrapper.blob[2]
				c0[y-s:y+s,x-s:x+s]=c00#np.max(C)
				self._myMainWindow._canvas_wrapper.blob=[]
			elapsed=time.time()-tini
			print('Max fps :',1/elapsed)
			#print(elapsed)
			if elapsed < dt/3600/24./self._fps:
				time.sleep( dt/3600/24./self._fps-elapsed)
			if self._should_end == True:
				break
		print("Exiting ...")
		self.stop_data()
		#self.finished.emit()


	def stop_data(self):
		print("Data source is quitting...")
		self._should_end = True

	def noise_corr(self,n,tmax,tcorr):
		V=np.random.randn(n,tmax)
		VF=fft(V,axis=0)
		k=2*np.pi*fftfreq(n, d=1.0/Np)
		VFf=[]
		for i in range(VF.shape[0]):
			tf=np.uint16(np.minimum(np.maximum(1/k[i]/2/np.pi*Np*tcorr,1),Np))
			filt=gaussian_filter(VF[i,:],tf,mode='wrap')
# 			win = signal.windows.hann(tf)
# 			filt=signal.convolve(VF[i,:], win, mode='wrap') / sum(win)
			VFf.append(filt/np.mean(np.abs(filt))*np.mean(np.abs(VF[i,:])))
		VFf=np.array(VFf)
		VFf[np.isnan(VFf)]=0
		return VFf
	
	def noise_corr_phi(self,tmax,tcorr):
		Phi=np.random.randn(tmax,Np,Np*p)
		PhiF=fft(fft(Phi,axis=1),axis=2)
		ky=2*np.pi*np.tile(fftfreq(Np, d=1.0/Np),(Np*p,1)).T
		kx=2*np.pi*np.tile(fftfreq(Np*p, d=1.0/Np),(Np,1))
		k=np.sqrt(ky**2+kx**2)
		PhiFf=np.copy(PhiF)
		for i in range(PhiF.shape[1]):
			for j  in range(PhiF.shape[2]):
				tf=np.uint16(np.minimum(np.maximum(1/k[i,j]/2/np.pi*Np*tcorr,1),tmax-1))
				filt=gaussian_filter(PhiFf[:,i,j],tf,mode='wrap')
				PhiFf[:,i,j]=filt
		return PhiFf


	def flow_1D(self,VF,flow=True,t=0):
		k=2*np.pi*fftfreq(len(VF), d=1.0/Np)
		K=k**(self._power*0.5)
		K[k<np.pi*2/self._lmax]=0
		K[k>np.pi*2/self._lmin]=0
		VFK=(VF.T*K).T
		VFK[np.isnan(VFK)]=0
		if self._mode=='Source': #shift by mean flow
			VFK=VFK*np.exp(1j*t*self._a/self._tcorr*2.0)
		v=np.real(np.fft.ifft(VFK,axis=0))
		return v/np.std(v)

	def flow_2D(self,PhiFf,flow=True,t=0):
		ky=2*np.pi*np.tile(fftfreq(Np, d=1.0/Np),(Np*p,1)).T
		kx=2*np.pi*np.tile(fftfreq(Np*p, d=1.0/Np),(Np,1))
		k=np.sqrt(ky**2+kx**2)
		K=k**(self._power-1)# -1 because it is the streamfunction( need to derive to get velocity)
		K[k<np.pi*2/self._lmax]=0
		K[k>np.pi*2/self._lmin]=0
		PhiFfk=PhiFf*K
		PhiFfk[np.isnan(PhiFfk)]=0
		vxf=PhiFfk*np.exp(-1j*ky)
		vyf=-PhiFfk*np.exp(-1j*kx)
		vx=np.real(ifft(ifft(vxf,axis=1),axis=0))
		vy=np.real(ifft(ifft(vyf,axis=1),axis=0))
		vn=np.sqrt(vx**2+vy**2)
		return vx/np.std(vn),vy/np.std(vn)

if __name__ == "__main__":
	app = use_app("pyqt5")
	app.create()

	canvas_wrapper = CanvasWrapper()
	win = MyMainWindow(canvas_wrapper)
	data_thread = QtCore.QThread(parent=win)
	data_source = DataSource(win)
	data_source.moveToThread(data_thread)

	# update the visualization when there is new data
	data_source.new_data.connect(canvas_wrapper.update_data)
	
	# start data generation when the thread is started
	data_thread.started.connect(data_source.run_data_creation)
	
	# if the data source finishes before the window is closed, kill the thread
	data_source.finished.connect(data_thread.quit, QtCore.Qt.DirectConnection)
	
	# if the window is closed, tell the data source to stop
	win.closing.connect(data_source.stop_data, QtCore.Qt.DirectConnection)
	
	# when the thread has ended, delete the data source from memory
	data_thread.finished.connect(data_source.deleteLater)

	win.show()
	data_thread.start()
	app.run()

	print("Waiting for data source to close gracefully...")
	data_thread.wait(5000)
