import imageio
import numpy as np

def analysis1():
    # 打开两个视频
    reader1 = imageio.get_reader('/home/cli7/CSGDR/nvdiffrast/debug/grads_all3.mp4')
    reader2 = imageio.get_reader('/home/cli7/CSGDR/nvdiffrast/resultdemo5.mp4')

    # 获取视频的 FPS
    fps = reader1.get_meta_data()['fps']

    # 创建一个 VideoWriter 对象来写入视频
    writer = imageio.get_writer('output_nall.mp4', fps=fps)
    import pdb
    for i, (frame1, frame2) in enumerate(zip(reader1, reader2)):
        # 找到第一个视频帧中的非黑色像素
        mask = np.where(np.sum(frame1,axis=-1)>10)
        
        # 使用第一个视频帧中的非黑色像素覆盖第二个视频帧中的对应像素
        frame2[mask] = [255,255,255]

        # 写入输出视频
        writer.append_data(frame2)

    # 释放资源
    writer.close()
def analysis2():
    #load a image
    img = imageio.imread('/home/cli7/CSGDR/nvdiffrast/target.png')
    #load the video
    reader = imageio.get_reader('/home/cli7/CSGDR/nvdiffrast/resultdemo5.mp4')
    # 获取视频的 FPS
    fps = reader.get_meta_data()['fps']

    # 创建一个 VideoWriter 对象来写入视频
    writer = imageio.get_writer('output_compare.mp4', fps=fps)
    import pdb
    for i, frame in enumerate(reader):
        #找到frame和img的差别
        pdb.set_trace()
        mask = np.where(np.sum(np.abs(frame-img),axis=-1)>200)
        # pdb.set_trace(  )
        # 使用第一个视频帧中的非黑色像素覆盖第二个视频帧中的对应像素
        frame[mask] = img[mask]*0.5

        # 写入输出视频
        writer.append_data(frame)
    
    
if __name__ == "__main__":
    analysis1()