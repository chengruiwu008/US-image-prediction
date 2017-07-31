from PIL import Image
def mergei(files, output_file):
    """
    横向拼接
    :param files:需要拼接的文件,list 
    :param output_file: 拼接完成后的输出文件
    :return：生成拼接后的新的图片 
    """
    tot = len(files)
    img = Image.open(files[0])
    w, h = img.size[0], img.size[1]
    merge_img = Image.new('RGB', (w * tot, h), 0xffffff)
    i = 0
    for f in files:
        print(f)
        img = Image.open(f)
        merge_img.paste(img, (i, 0))
        i += w
    merge_img.save(output_file)

for i in range(5,353):
    files = ['./Bruce_list_1_sentence_50_9696/%d.jpg' % (i), './pred_result_challenge/%d.jpg' % i]
    output_file = './pred_result_challenge_combine/%d.jpg' % i
    mergei(files, output_file)