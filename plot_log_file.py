import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("log.out","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            _, x, _, y = eachLine.strip().split(' ')
            xar.append(int(x))
            yar.append(int(y))
    ax1.clear()
    ax1.scatter(xar,yar)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
