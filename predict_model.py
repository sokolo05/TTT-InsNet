import pysam
import numpy as np
import math
import os
import torch
from train_model import TTT_InsNet
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

def decode_flag(Flag):

    signal = {1 << 2: 0, 1 >> 1: 1, 1 << 4: 2, 1 << 11: 3, 1 << 4 | 1 << 11: 4}

    return signal[Flag] if(Flag in signal) else 0

def c_pos(cigar, refstart):

    number = ''
    numlist = [str(i) for i in range(10)]
    readstart = False
    readend = False
    refend = False
    readloc = 0
    refloc = refstart
    for c in cigar:
        if(c in numlist):
            number += c
        else:
            number = int(number)
            if(readstart == False and c in ['M', 'I', '=', 'X']):
                readstart = readloc
            if(readstart != False and c in ['H', 'S']):
                readend = readloc
                refend = refloc
                break
            if(c in ['M', 'I', 'S', '=', 'X']):
                readloc += number
            if(c in ['M', 'D', 'N', '=', 'X']):
                refloc += number
            number = ''
    if(readend == False):
        readend = readloc
        refend = refloc

    return refstart, refend, readstart, readend 

def ins_signature(pre, bamfile):

    data = []
    for chr_name,start,end in pre:

        for read in bamfile.fetch(chr_name,start,end):
            aligned_length = read.reference_length
            if aligned_length == None:
                aligned_length= 0
            if (read.mapping_quality >= 0) and aligned_length >= 0:
                cigar = []
                sta = read.reference_start
                for ci  in read.cigartuples:
                    if ci[0] in [0, 2, 3, 7, 8]:
                        sta += ci[1]
                    elif ci[0] == 1 :
                        if ci[1] >=50 and (abs(sta-start) < 200):
                            cigar.append([sta,sta,ci[1]])
                if len(cigar) !=0:
                    cigar = np.array(cigar)
                    cigar = cigar[np.argsort(cigar[:,0])]
                    a = mergecigar(cigar)
                    data.extend(a)
            if(read.has_tag('SA') == True):
                code = decode_flag(read.flag)
                rawsalist = read.get_tag('SA').split(';')
                for sa in rawsalist[:-1]:
                    sainfo = sa.split(',')
                    tmpcontig, tmprefstart, strand, cigar = sainfo[0], int(sainfo[1]), sainfo[2], sainfo[3]
                    if(tmpcontig != chr_name):
                        continue 
                    if((strand == '-' and (code %2) ==0) or (strand == '+' and (code %2) ==1)):
                        refstart_1, refend_1, readstart_1, readend_1 =  read.reference_start, read.reference_end,read.query_alignment_start,read.query_alignment_end
                        refstart_2, refend_2, readstart_2, readend_2 = c_pos(cigar, tmprefstart)
                        a = readend_1 - readstart_2
                        b = refend_1 - refstart_2
                        if(abs(b-a)<30):
                            continue
                        if((b-a)>=50 and ((b-a)>0)):
                            data22 = []                          
                            if(refend_1<=end and refend_1>=start):
                                data22.append([refend_1,refend_1,abs((b-a))])
                            if(refstart_2<=end and refstart_2>=start):
                                data22.append([refstart_2,refstart_2,abs((b-a))])
                            data22 = np.array(data22)
                            if len(data22)==0:
                                continue
                            data.extend(data22)    
    data = np.array(data)
    if len(data) == 0:
        return data
    data = data[np.argsort(data[:,0])]
                      
    return data

def mergecigar(infor):

    data = []
    i = 0
    while i>=0:
        count = 0
        if i >(len(infor)-1):
            break
        lenth = infor[i][2]
        for j in range(i+1,len(infor)):
            if abs(infor[j][1] - infor[i][1]) <= 40: 
                count = count + 1
                infor [i][1] = infor[j][0]#改[0]0
                lenth = lenth +  infor[j][2] #+ abs(infor[j][0] - infor[i][0])
        data.append([infor[i][0],infor[i][0]+1, lenth])
        if count == 0:
            i += 1
        else :
            i += (count+1)

    return data

def merge(infor):

    data = []
    i = 0
    while i>=0:
        dat = []     
        count = 0
        if i >(len(infor)-1):
            break
        dat.append(infor[i])
        for j in range(i+1,len(infor)):
            if( (abs(infor[i][0] -infor[j][0]) <= 1500) and (abs(infor[i][1] - infor[j][1])<= 1500)):
                count = count + 1
                dat.append(infor[j])
        dat = np.array(dat)
        data.append(dat)
        if count == 0:
            i += 1
        else :
            i += (count+1)

    return data

def merge_insnet_long(pre, index, chr_name, bamfile):

    data = []
    insertion = []
    for i in range(len(pre)):
        if pre[i] > 0.5:
            data.append([chr_name, index[i], index[i] + 200])
    signature = ins_signature(data, bamfile)
    ins_sigs = merge(signature)
    for sig in ins_sigs:
            pp = np.array(sig)
            start = math.ceil(np.median(pp[:, 0]))
            kk = int(len(pp)/2)
            svle = np.sort(pp[:,2])
            length = math.ceil(np.median(svle[kk:]))
            insertion.append([chr_name, start, length, len(pp), 'INS'])

    return insertion 

def predict_step(base, predict):

    for i in range(len(predict)):
        if predict[i] >= 0.5:
            # print(f'predict[i] = {predict[i]}')
            base[i] = 1
            base[i + 1] = 1

    return base

def tovcf(sv_callers, contig2length, sv_types, outvcfpath, version):

    vcf_output = open(outvcfpath, 'w')
    
    print("##fileformat=VCFv4.2", file=vcf_output)
    print("##fileDate={0}".format(time.strftime("%Y-%m-%d|%I:%M:%S%p|%Z|%z")), file=vcf_output)
    print("##source=INSnet_pro-v{0}".format(version), file=vcf_output)
    print("##FILTER=<ID=PASS,Description=\"All filters passed\">", file=vcf_output)
    for contig, length in contig2length.items():
        print("##contig=<ID={0},length={1}>".format(contig, length), file=vcf_output)
        
    if "INS" in sv_types:
        print("##ALT=<ID=INS,Description=\"Insertion\">", file=vcf_output)
    if "DEL" in sv_types:
        print("##ALT=<ID=DEL,Description=\"Deletion\">", file=vcf_output)
    if "INV" in sv_types:
        print("##ALT=<ID=INV,Description=\"Inversion\">", file=vcf_output)
    if "BND" in sv_types:
        print("##ALT=<ID=BND,Description=\"Breakend\">", file=vcf_output)
        
    print("##INFO=<ID=END,Number=1,Type=Integer,Description=\"End position of the structural variant\">", file=vcf_output)
    print("##INFO=<ID=SVTYPE,Number=1,Type=String,Description=\"Type of structural variant\">", file=vcf_output)
    print("##INFO=<ID=SVLEN,Number=1,Type=Integer,Description=\"Difference in length between REF and ALT alleles\">", file=vcf_output)
    print("##INFO=<ID=SUPPORT,Number=1,Type=Integer,Description=\"Number of read support this record\">", file=vcf_output)
    print("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">", file=vcf_output)
    print("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t.", file=vcf_output)
    
    ins_id = 0
    for sv in sv_callers:
        if sv[4] == 'INS':
            recinfo = 'SVLEN=' + str(int(sv[2])) + ';SVTYPE=' + str(sv[4]) + ';END=' + str(sv[1]) + ';SUPPORT=' + str(sv[3]) + '\tGT\t' + '.\n'
        vcf_output.write(sv[0] + '\t' + str(int(sv[1])) + '\t' + 'INS-pro.'+ str(ins_id) + '\t' + 'N' + '\t' + '<' + str(sv[4]) + '>' + '\t' + str(int(sv[3])+1) + '\t' + 'PASS' + '\t' + recinfo)
        ins_id += 1
    vcf_output.close()

def batchdata(data, timesteps, step, num_gpus = 2, window = 200):

    if step != 0:
        data = data.reshape(-1, 5)[step:(step - window)]
    data = data.reshape(-1, 200, 5)

    size = data.shape[0] // (timesteps * num_gpus)
    size_ = data.shape[0] % (timesteps * num_gpus)

    return data[: size * (timesteps * num_gpus)], data[size * (timesteps * num_gpus) :]

def process_data(data, batch_size, timesteps, offset, num_gpus, predict_ins, device):

    # print(f'data = {data.shape}')
    features_main, features_remain = batchdata(data, timesteps, offset)
    features_main = features_main.reshape(-1, 200, 5, 1) if len(features_main) > 0 else np.array([])
    features_remain = features_remain.reshape(-1, 200, 5, 1) if len(features_remain) > 0 else np.array([])
    # print(f'features_main = {features_main.shape}, features_remain = {features_remain.shape}')

    remain = features_remain.shape[0] % (timesteps * num_gpus)
    padding_size = (timesteps * num_gpus) - remain
    # print(f'remain = {remain}, padding_size = {padding_size}')

    if remain != 0:
        patch_features = np.zeros((padding_size, 200, 5, 1))
        # print(f'patch_features = {patch_features.shape}, features_remain = {features_remain.shape}, features_main = {features_main.shape}')
        features_padded = np.concatenate((features_remain, patch_features), axis=0)
        if len(features_main) > 0:
            features = np.concatenate((features_main, features_padded), axis=0)
        else:
            features = features_padded
    else:
        features = features_main
    
    # print(f'features = {features.shape}')
    predict = predict_data(features, predict_ins, device, batch_size, timesteps)
    # print(f'predict = {predict.shape}')
    if len(predict) == 0:
        return np.array([])
    else:
        predict_main = predict[:-1, :, :]
        predict_remain =  predict[-1:, :, :]
        # print(f'predict_main = {predict_main.shape}, predict_remain = {predict_remain.shape}')
        predict_main = predict_main.flatten()
        predict_remain = predict_remain.flatten()
        if padding_size == timesteps * num_gpus:
            predict_remain = predict_remain[:]
        else:
            predict_remain = predict_remain[:-padding_size]
        # print(f'predict_main = {predict_main.shape}, predict_remain = {predict_remain.shape}')
        predict_end = np.concatenate((predict_main, predict_remain), axis=0)

        return predict_end

def predict_data(feature, predict_ins, device, batch_size, timesteps):

    if len(feature) == 0:
        return np.array([])
    if isinstance(feature, np.ndarray):
        feature = torch.tensor(feature, dtype=torch.float32)

    dataset = TensorDataset (feature)
    dataloader = DataLoader(dataset, batch_size=batch_size * timesteps, shuffle=False)

    predict_ins.eval()
    predictions = []

    with torch.no_grad():  # 关闭梯度计算
        for batch in dataloader:
            inputs = batch[0].to(device)  # 确保数据在正确的设备上
            # print(f'inputs no_grad = {inputs.shape}')
            outputs = predict_ins(inputs)  # 模型预测
            predictions.append(outputs.cpu().numpy())  # 将预测结果转换为 numpy 数组并存储

    # 将所有预测结果合并为一个数组
    return np.concatenate(predictions, axis=0)

def predict_funtion(gpu_name, save_length, timesteps, ins_predict_weight, data_path, bam_file, out_vcf_file, contigs, support):

    print(f'gpu_name = {gpu_name}, type of gpu_name = {type(gpu_name)}')
    # gpu_name_str = ','.join(map(str, gpu_name)) if gpu_name else '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bamfile_data = pysam.AlignmentFile(bam_file,'rb')
    index_stats = bamfile_data.get_index_statistics()
    contigs_lengths = {stat.contig: bamfile_data.lengths[i] for i, stat in enumerate(index_stats)}
    save_length, timesteps, support = int(save_length), int(timesteps), int(support)
    resultlist = [['CONTIG', 'START', 'SVLEN', 'READ_SUPPORT', 'SVTYPE']]

    predict_ins = TTT_InsNet(timesteps)
    state_dict = torch.load(ins_predict_weight)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    predict_ins.load_state_dict(new_state_dict, strict=False)
    # predict_ins = nn.DataParallel(predict_ins)
    predict_ins.to(DEVICE)
    
    print(f'contigs = {contigs}, type of contigs = {type(contigs)}')
    for chr_name in contigs:
        print(f'chr_name = {chr_name}')
        chr_name = str(chr_name)
        chr_name = chr_name.replace('chr', '') if 'chr' in chr_name else chr_name
        chr_length = contigs_lengths[str(chr_name)]
        iders = math.ceil(chr_length / save_length)
        start = 0
        print(f'+++++++ chr_name = {chr_name}, iders = {iders} ++++++++++')
        for ider in range(iders):
            print(f'insertion_predict_chr : chr_name = {chr_name}, ider / iders = {ider} / {iders}')
            try:
                data_name = data_path +'/chr'+ chr_name + '_' + str(start)  +  '_' + str(start + save_length) + '.npy'
                data = np.load(data_name)
                data = data[:, :-1]
                print(f'data = {data.shape}')
            except FileNotFoundError:
                print(f'File not found: {data_name}, skipping this segment.')
                start = start + save_length
                continue
            else:
                index_name = data_path +'/chr'+ chr_name + '_' + str(start)  +  '_' + str(start + save_length) + '_index.npy'
                index  = np.load(index_name)
                if len(data) == 0:
                    continue
                print(f'data = {data.shape}, index = {index.shape}')
                base = process_data(data, 20, 100, 0, 2, predict_ins, DEVICE) # 0
                base = predict_step(base, process_data(data, 20, 100, 100, 2, predict_ins, DEVICE)) # 2
                base = predict_step(base, process_data(data, 20, 100, 50, 2, predict_ins, DEVICE)) # 1
                base = predict_step(base, process_data(data, 20, 100, 150, 2, predict_ins, DEVICE)) # 3
 
                contig, start = chr_name, start
                resultlist += merge_insnet_long(base, index, contig, bamfile_data)

                start = start + save_length
    sv_callers = []
    for read in resultlist[1:]:
        if read[3] >= int(support) and read[2] >= 50: 
            sv_callers.append([read[0], read[1], read[2], read[3], 'INS', '.'])
    sv_types = ['INS']

    tovcf(sv_callers, contigs_lengths, sv_types, out_vcf_file, version=1)
    
    return resultlist