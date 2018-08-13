import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import psutil
import time

#plt.style.use('ggplot')   # othre options #fivethirtyeight   #dark_background #bmh
#plt.style.use('dark_background')
fig=plt.figure(1)

ax1=fig.add_subplot(2,2,1)					# subplots represented in different plot style
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4,projection='3d')

ls1=[]
ls2=[]
 
def animate(i):
	[ls1.append(time.monotonic())]				# appending monotonic time to list
	[ls2.append(psutil.virtual_memory()[2])]	# appending CPU usage percentage to a list
	
	ax1.plot(ls1,ls2)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('CPU usage')
	ax1.grid(True)
							#  other options include ax.hist2d(ls1,ls2)  ax.scatter(ls1, ls2)
	ax2.scatter(ls1,ls2)
	ax2.set_xlabel('Time')
	ax2.set_ylabel('CPU usage')

	ax3.hist2d(ls1,ls2)
	ax3.set_xlabel('Time')
	ax3.set_ylabel('CPU usage')
	
	ax4.plot(ls1,ls2)
	ax4.set_xlabel('Time')
	ax4.set_ylabel('CPU usage')

ani=animation.FuncAnimation(fig, animate, interval=100)
plt.show()
