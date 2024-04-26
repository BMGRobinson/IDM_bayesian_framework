from numpy import sin, cos, pi, array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from JSAnimation import HTMLWriter

dt = 0.01
t = np.arange(0.0, 30, dt)

#Load airfoil points
coords = np.loadtxt('naca0012h_coords.dat')
Xinitial = coords[:,0]
Yinitial = coords[:,1]

#Load signal
signal = np.loadtxt('signal.dat')

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.0, 2), ylim=(-2, 2))
ax.grid(False)

line, = ax.plot([], [], lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
	line.set_data([], [])
	time_text.set_text('')
	return line,time_text

def animate(i):
	h = signal[i,0]
	theta = signal[i,2]
	
	#Rotation about the center point
	xc = 0.6
	yc = 0
	
	#Rotation
	deltaX = Xinitial - xc
	deltaY = Yinitial - yc
	
	X = xc + deltaX * cos(theta) - deltaY * sin(theta)
	Y = yc + deltaX * sin(theta) + deltaY * cos(theta)
	
	#Heave translation
	Y = Y - h
	
	line.set_data(X, Y)
	time_text.set_text(time_template%(i*dt))
	return line, time_text

animation.FuncAnimation(fig, animate, frames = 200, interval=10, blit=True, init_func=init)
#return animation.FuncAnimation(fig, animate, np.arange(1, len(t)), interval=10, blit=True, init_func=init)
#anim.save('airfoil_sim.html', writer=HTMLWriter(embed_frames=True))