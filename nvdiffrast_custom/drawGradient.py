source_file = '/home/yyyyyhc/nvdiffrast/exp/vis_grad_v2_no_intersection_color_hyperparam_0.05_1000/random/cubeSphere/t_0/delta_1.npy'
source_file2 = '/home/yyyyyhc/nvdiffrast/exp/vis_grad_v2_with_intersection_color_hyperparam_0.05_1000/random/cubeSphere/t_0/delta_1.npy'

source_fd = '/home/yyyyyhc/nvdiffrast/exp/vis_grad_v2_no_intersection_color_hyperparam_0.05_1000/random/cubeSphere/t_0/delta_fdrev_0.01.npy'
import numpy as np
import pdb
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def get_color_stripe(colors):
    

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(2, 10))

    # Display the color stripe as an image with no horizontal variation
    ax.imshow(np.expand_dims(colors, axis=1), aspect='auto', extent=[0, 1, -1, 1])

    # Remove x-ticks and customize y-ticks
    ax.axis('off')

    # Save the figure without displaying it
    fig.savefig('./Color_Stripe.png', bbox_inches='tight')

from PIL import Image, ImageFilter, ImageChops

def dilate_image(image_path, output_path, dilation_size=1):
    # 打开图像
    image = Image.open(image_path).convert("RGBA")
    
    # 提取alpha通道
    alpha = image.split()[-1]
    
    # 使用膨胀滤镜
    dilation_kernel = ImageFilter.Kernel((3, 3), [1] * 9, 1, 0)
    dilated_alpha = alpha.filter(ImageFilter.MaxFilter(size=2 * dilation_size + 1))
    
    # 用修改后的alpha通道替换原来的alpha通道
    new_image = ImageChops.composite(image, image, dilated_alpha)
    
    # 保存图片
    new_image.save(output_path)

def mylog(a, b):
    print(a,b)
    return np.log(a+1e-5) / (np.log(b+1e-5)+1e-5)

def value_to_hsv(value):
    """
    Maps a value in the range [-1, 1] to an HSV color.
    -1 -> blue-purple (hue = 270 degrees)
     0 -> yellow-green (hue = 90 degrees)
     1 -> orange-red (hue = 0 degrees)
    """
    # Normalize value to range [0, 1] from [-1, 1]
    normalized = (value + 1) / 2
    # Linear interpolation between hue values
    hue = 270 * (1 - normalized) + 0 * normalized  # from blue-purple to orange-red through yellow-green
    
    rgb = mcolors.hsv_to_rgb((hue / 360, 1, 1))  # full saturation and value
    alpha = np.abs(value)
    rgba = np.append(rgb, alpha)
    return rgba  # full saturation and value

all_grads = np.load(source_file, allow_pickle=True)
all_grads_2 = np.load(source_file2, allow_pickle=True)
all_grads_fd = np.load(source_fd, allow_pickle=True)
#visualize difference
diff_1 = all_grads - all_grads_fd
diff_2 = all_grads_2 - all_grads_fd

diff_intensity_1 = np.sum(np.abs(diff_1), axis=2)
diff_intensity_2 = np.sum(np.abs(diff_2), axis=2)

minFactor = np.min(np.array([diff_intensity_1.min(), diff_intensity_2.min()]))
maxFactor = np.max(np.array([diff_intensity_1.max(), diff_intensity_2.max()]))

scaled_intensity_1 = ((diff_intensity_1- minFactor)/(maxFactor-minFactor))*255
scaled_intensity_2 = ((diff_intensity_2- minFactor)/(maxFactor-minFactor))*255

scaled_intensity_1 = np.where(scaled_intensity_1 > 255, 255, scaled_intensity_1)
scaled_intensity_2 = np.where(scaled_intensity_2 > 255, 255, scaled_intensity_2)

scaled_intensity_1 = np.where(scaled_intensity_1 < 0, 0, scaled_intensity_1)
scaled_intensity_2 = np.where(scaled_intensity_2 < 0, 0, scaled_intensity_2)

imageio.imwrite('diff_intensity_1.png', (scaled_intensity_1).astype(np.uint8))
imageio.imwrite('diff_intensity_2.png', (scaled_intensity_2).astype(np.uint8))

# all_grads = all_grads_fd.copy()
pdb.set_trace()
grad_mask = np.where(all_grads !=0, 1, 0)
grad_intensity = np.sum(np.abs(all_grads), axis=2)
scaled_intensity =((grad_intensity- np.min(grad_intensity))/(np.max(grad_intensity)-np.min(grad_intensity)))*255
all_grads_c1 = (all_grads[:,:,0] + all_grads[:,:,1] + all_grads[:,:,2])

maxC1 = all_grads_c1.max()
minC1 = all_grads_c1.min()
boost = 1.1
all_grads_c1 = np.where(all_grads_c1 >0, mylog(1+all_grads_c1, boost)/(mylog(1+np.max(all_grads_c1),boost)), -1 * mylog(1-all_grads_c1,boost)/mylog(1-np.min(all_grads_c1),boost))
all_grads_c1_binary = np.abs(all_grads_c1)
all_grads_c1_binary = np.log(1+all_grads_c1_binary)/np.log(1+np.max(all_grads_c1_binary))
all_grads_c1_image = np.zeros((all_grads_c1.shape[0], all_grads_c1.shape[1], 4))
for i in range(all_grads_c1.shape[0]):
    for j in range(all_grads_c1.shape[1]):
        all_grads_c1_image[i,j] = value_to_hsv(all_grads_c1[i,j])
        
all_grads_c2 = (all_grads_2[:,:,0] + all_grads_2[:,:,1] + all_grads_2[:,:,2])
boost = 1.1
all_grads_c2 = np.where(all_grads_c2 >0, mylog(1+all_grads_c2, boost)/(mylog(1+maxC1,boost)), -1 * mylog(1-all_grads_c2,boost)/mylog(1-np.min(minC1),boost))
all_grads_c2_binary = np.abs(all_grads_c2)
all_grads_c2_binary = np.log(1+all_grads_c2_binary)/np.log(1+np.max(all_grads_c2_binary))
all_grads_c2_image = np.zeros((all_grads_c2.shape[0], all_grads_c2.shape[1], 4))
for i in range(all_grads_c2.shape[0]):
    for j in range(all_grads_c2.shape[1]):
        all_grads_c2_image[i,j] = value_to_hsv(all_grads_c2[i,j])
# Generate values from -1 to 1 with an interval of 0.01
values = np.arange(-10, 10, 0.01)
boost = 1.1
# Generate colors for each value using the get_color function
values = np.where(values >0, mylog(1+values, boost)/(mylog(1+maxC1,boost)), -1 * mylog(1-values,boost)/mylog(1-np.min(minC1),boost))

colors = np.array([value_to_hsv(val) for val in values])
get_color_stripe(colors)
# pdb.set_trace()

all_grads_fd = (all_grads_fd[:,:,0] + all_grads_fd[:,:,1] + all_grads_fd[:,:,2])
boost = 1.1
all_grads_fd = np.where(all_grads_fd >0, mylog(1+all_grads_fd, boost)/(mylog(1+maxC1,boost)), -1 * mylog(1-all_grads_fd,boost)/mylog(1-np.min(minC1),boost))
all_grads_fd_binary = np.abs(all_grads_fd)
all_grads_fd_binary = np.log(1+all_grads_fd_binary)/np.log(1+np.max(all_grads_fd_binary))
all_grads_fd_image = np.zeros((all_grads_fd.shape[0], all_grads_fd.shape[1], 4))
for i in range(all_grads_fd.shape[0]):
    for j in range(all_grads_fd.shape[1]):
        all_grads_fd_image[i,j] = value_to_hsv(all_grads_fd[i,j])
imageio.imwrite('grad_mask_colorfd.png', (all_grads_fd_image*255).astype(np.uint8))
imageio.imwrite('grad_mask_binaryfd.png', (all_grads_fd_binary*255).astype(np.uint8))
# pdb.set_trace()
imageio.imwrite('grad_mask_color1.png', (all_grads_c1_image*255).astype(np.uint8))
imageio.imwrite('grad_mask_binary1.png', (all_grads_c1_binary*255).astype(np.uint8))

imageio.imwrite('grad_mask_color2.png', (all_grads_c2_image*255).astype(np.uint8))
imageio.imwrite('grad_mask_binary2.png', (all_grads_c2_binary*255).astype(np.uint8))
# # 使用函数
# dilate_image('grad_mask.png', 'output_grad_mask.png', dilation_size=)

pdb.set_trace()
