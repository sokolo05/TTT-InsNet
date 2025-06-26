# TTT-InsNet
a novel time-distributed dual-transformer algorithm for large-scale insertions detection in long-read

# Installation

## Requirements
- python=3.9.1
- torch=2.5
- pandas=2.2.3
- pysam=0.23.0
- torchvision=0.20.0
- cuda=12.4

## 1. Create a virtual environment

```
#create
conda create -n your_environment_name python=3.9

#activate
conda activate your_environment_name
```

## 2. Package installation

```
pip install -r requirements.txt
```

## 3. Download TTT-InsNet code

```
git clone https://github.com/eioyuou/TTT-InsNet.git
cd TTT-InsNet
```
# Usage

## 1.Produce data for call SV

```
python TTT_InsNet.py generate_feature bam_file output_path contigs_list(default:[](all chromosomes)) max_worker vcf_file
    
bam_file:the path of the alignment file about the reference and the long read set;    
output_path:a folder which is used to store generated features data;  
contigs_list:the list of contig to preform detection.(default: [], all contig are used);
max_work:the number of threads to use;
vcf_file:the gold standard file for standard data.
   
eg:# python TTT_InsNet.py your_file_address/HG002_PB_5x_RG_HP10XtrioRTG.bam your_file_address/chr12-13 [12,13] 5 your_file_address/HG002_SVs_Tier1_v0.6.vcf.gz
```

## 2.Call insertion

```
python TTT_InsNet.py gpu_name save_length timesteps ins_predict_weight data_path bam_file out_vcf_file contigs support

gpu_name:num of the GPU to use;
save_length:the feature file spans across nucleotide base sequence lengths;
timesteps:time step of time-distributed;
ins_predict_weight:path of insert predict weight file;
data_path:a folder for storing evaluation feature files;
bam_file:path of the alignment file about the reference and the long read set;
out_vcf_file:the path of output vcf file;
contigs:the list of contig to preform detection.(default: [], all contig are used);
support:min support reads.
   
python TTT_InsNet.py call_insertion '1, 2' 10000000 100 your_file/ins_predict_weight.pth your_file/data_path your_file/bam_file.bam your_file/out_vcf_file.vcf [12,13] 5
```

# Tested data

## HG002 CLR data
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/Baylor_NGMLR_bam_GRCh37/HG002_PB_70x_RG_HP10XtrioRTG.bam
```

## HG002 ONT data
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/UCSC_Ultralong_OxfordNanopore_Promethion/HG002_GRCh37_ONT-UL_UCSC_20200508.phased.bam
```

## HG002 CCS data
```
https://ftp.ncbi.nih.gov/giab/ftp/data/AshkenazimTrio/HG002_NA24385_son/PacBio_CCS_15kb/alignment/HG002.Sequel.15kb.pbmm2.hs37d5.whatshap.haplotag.RTG.10x.trio.bam
```
