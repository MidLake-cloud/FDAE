import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

 
def process_image(text, title, i, source_idx, source_content, recon_content, target_idx, target_content, label, save_path, model_name="FDAe"):
    fig = plt.Figure(figsize=(12, 4), dpi=50)
    gcf = plt.gcf()
    # gcf.set_size_inches(15, 7)
    gcf.set_size_inches(12, 7)
    fig.set_canvas(gcf.canvas)
    height = 5
    width = 1
    plt.subplot(height, width, 1)
    plt.plot(source_content)
    plt.title(f"[{text}] input: {title}, source idx: {source_idx}, label {label}")

    plt.subplot(height, width, 2)
    plt.plot(recon_content)
    plt.title(f"reconstruction, source idx {source_idx}, label {label}")

    if target_idx != None:
        plt.subplot(height, width, 3)
        plt.plot(target_content)
        plt.title(f"generation, target idx {target_idx}, label {label}")
        save_file = f"ecg_img_{text}_{source_idx} to {target_idx}_{i}_class {label}.png"
    else:
        save_file = f"ecg_img_{text}_{source_idx}_{i}_class {label}.png"

    plt.tight_layout()
    # plt.show()

    img_save_path = os.path.join(save_path, "imgs")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    fig.savefig(os.path.join(img_save_path, save_file), bbox_inches="tight", pad_inches=0)
    # fig.savefig(prepath+f'{model_name}-images/image-' + text + "-" + str(target_idx) + "-" + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f'source patient: {source_idx}, target patient: {target_idx}, label: {label}, saving path: {img_save_path}')


def process_ecg_cvae(text, title, i, fake_content, label, save_path):
    print(f'ecg label: {label}')
    fig = plt.Figure(figsize=(12, 4), dpi=50)
    gcf = plt.gcf()
    # gcf.set_size_inches(15, 7)
    gcf.set_size_inches(12, 7)
    fig.set_canvas(gcf.canvas)
    height = 2
    width = 1
    plt.subplot(height, width, 1)
    plt.plot(fake_content)
    plt.title(f"[{text}] {title}, label {label}")

    plt.tight_layout()
    # plt.show()

    img_save_path = os.path.join(save_path, "imgs")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    fig.savefig(os.path.join(img_save_path, f"ecg_image_{text}_{i}_class {label}.png"), bbox_inches="tight", pad_inches=0)
    # fig.savefig(prepath+f'{model_name}-images/image-' + text + "-" + str(target_idx) + "-" + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def visualization_ecgs(k, mode, content_list, label_list, patient_list, intro_list, save_path):
    fig = plt.Figure(figsize=(8, 4), dpi=50)
    gcf = plt.gcf()
    gcf.set_size_inches(12, 16)
    fig.set_canvas(gcf.canvas)
    ecg_nums = len(content_list)
    height = 1 + ecg_nums
    width = 2
    for idx, (ecg, label, patient, intro) in enumerate(zip(content_list, label_list, patient_list, intro_list)):
        plt.subplot(height, width, idx+1)
        plt.plot(ecg)
        plt.title(f"[{mode}] {intro}, label {label}, patient {patient}")
    plt.tight_layout(pad=2)
    save_file = f"ecg_img_{mode}_{k}.png"
    img_save_path = os.path.join(save_path, "imgs")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    fig.savefig(os.path.join(img_save_path, save_file), bbox_inches="tight", pad_inches=0.3)
    print("Saved in path {}".format(os.path.join(img_save_path, save_file)))
    # plt.show()
    plt.close()


def draw_loss(values, title, save_path):
    plt.title(title)
    plt.plot(values)
    plt.savefig(os.path.join(save_path, f'{title}.png'))
    plt.close()


def draw_macro_pr(file_root, model_name):
    import pandas as pd
    file_name = '{}/pred_result.csv'.format(file_root)
    content: np.ndarray = pd.read_csv(file_name).values # [N, 2+4]
    print('drawing PR curve of {}'.format(file_name))
    lis_all = content[:, -4:].reshape(-1).tolist()
    lis_order = sorted(set(lis_all))

    macro_precis = []
    macro_recall = []

    for i in lis_order:

        true_p0 = 0         
        true_n0 = 0         
        false_p0 = 0        
        false_n0 = 0        

        true_p1 = 0
        true_n1 = 0
        false_p1 = 0
        false_n1 = 0

        true_p2 = 0
        true_n2 = 0
        false_p2 = 0
        false_n2 = 0

        true_p3 = 0
        true_n3 = 0
        false_p3 = 0
        false_n3 = 0

        for file in content:
            cls, pred, n0, n1, n2, n3 = file.tolist()       
                                                

            if float(n0) >= float(i) and cls == 0:             
                true_p0 = true_p0 + 1                     
            elif float(n0) >= float(i) and cls != 0:           
                false_p0 = false_p0 + 1                   
            elif float(n0) < float(i) and cls == 0:
                false_n0 = false_n0 + 1

            if float(n1) >= float(i) and cls == 1:          
                true_p1 = true_p1 + 1
            elif float(n1) >= float(i) and cls != 1:
                false_p1 = false_p1 + 1
            elif float(n1) < float(i) and cls == 1:
                false_n1 = false_n1 + 1

            if float(n2) >= float(i) and cls == 2:            
                true_p2 = true_p2 + 1
            elif float(n2) >= float(i) and cls != 2:
                false_p2 = false_p2 + 1
            elif float(n2) < float(i) and cls == 2:
                false_n2 = false_n2 + 1

            if float(n3) >= float(i) and cls == 3:         
                true_p3 = true_p3 + 1
            elif float(n3) >= float(i) and cls != 3:
                false_p3 = false_p3 + 1
            elif float(n3) < float(i) and cls == 3:
                false_n3 = false_n3 + 1

        prec0 = (true_p0+0.00000000001) / (true_p0 + false_p0 + 0.00000000001) 
        prec1 = (true_p1+0.00000000001) / (true_p1 + false_p1 + 0.00000000001)
        prec2 = (true_p2+0.00000000001) / (true_p2 + false_p2 + 0.00000000001)
        prec3 = (true_p3+0.00000000001) / (true_p3 + false_p3 + 0.00000000001)

        recall0 = (true_p0+0.00000000001)/(true_p0+false_n0 + 0.00000000001)    
        recall1 = (true_p1+0.00000000001) / (true_p1 + false_n1+0.00000000001)
        recall2 = (true_p2+0.00000000001)/(true_p2+false_n2 + 0.00000000001)
        recall3 = (true_p3+0.00000000001)/(true_p3+false_n3 + 0.00000000001)

        precision = (prec0 + prec1 + prec2 + prec3) / 4
        recall = (recall0 + recall1 + recall2 + recall3) / 4   
        macro_precis.append(precision)
        macro_recall.append(recall)

    macro_precis.append(1)
    macro_recall.append(0)
    print(macro_precis)
    print(macro_recall)

    x = np.array(macro_recall)
    y = np.array(macro_precis)
    plt.figure()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PR curve of {}'.format(model_name))
    plt.plot(x, y)
    plt.show()
    plt.savefig(os.path.join(file_root, 'PR curve.png'))
    plt.close()