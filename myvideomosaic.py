import matplotlib.pyplot as plt
import matplotlib.animation as animation

def myvideomosaic(img_mosaic):
    plt.rcParams['animation.ffmpeg_path'] = './ffmpeg'

    ims = []
    fig = plt.figure()
    plt.axis('off')
    for i in range(len(img_mosaic)):
        im = plt.imshow(img_mosaic[i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, blit=True)

    myWriter = animation.FFMpegWriter()
    ani.save('mosaic.avi', writer=myWriter)
    plt.show()