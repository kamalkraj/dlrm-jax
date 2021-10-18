# dlrm-jax

### Download data

```bash
wget http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
md5sum criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz
# df9b1b3766d9ff91d5ca3eb3d23bed27
mkdir data
tar -xzf criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz -C data
```

### Run code 
#### CPU
```bash
./run_kaggle.sh
```
sample log:
```
Using CPU...
Reading processed data
Sparse fea = 26, Dense fea = 13
Defined train indices...
Randomized indices across days ...
Split data according to indices...
Reading processed data
Sparse fea = 26, Dense fea = 13
Defined test indices...
Randomized indices across days ...
Split data according to indices...
time/loss/accuracy (if enabled):
Finished training it 1024/306969 of epoch 0, 10.81 ms/it, loss 0.518778
Finished training it 2048/306969 of epoch 0, 10.42 ms/it, loss 0.504306
Finished training it 3072/306969 of epoch 0, 10.38 ms/it, loss 0.498378
Finished training it 4096/306969 of epoch 0, 10.54 ms/it, loss 0.487422
Finished training it 5120/306969 of epoch 0, 10.49 ms/it, loss 0.486536
Finished training it 6144/306969 of epoch 0, 10.32 ms/it, loss 0.481590
Finished training it 7168/306969 of epoch 0, 10.27 ms/it, loss 0.477130
Finished training it 8192/306969 of epoch 0, 10.56 ms/it, loss 0.476602
Finished training it 9216/306969 of epoch 0, 10.48 ms/it, loss 0.474181
Finished training it 10240/306969 of epoch 0, 10.49 ms/it, loss 0.471943
Testing at - 10240/306969 of epoch 0,
 accuracy 77.430 %, best 77.430 %
```
Tested CPU
```
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              32
On-line CPU(s) list: 0-31
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Core(TM) i9-9960X CPU @ 3.10GHz
Stepping:            4
CPU MHz:             1200.157
CPU max MHz:         4500.0000
CPU min MHz:         1200.0000
BogoMIPS:            6199.99
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            22528K
NUMA node0 CPU(s):   0-31
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti ssbd mba ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm mpx rdt_a avx512f avx512dq rdseed adx smap clflushopt clwb intel_pt avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req md_clear flush_l1d
```
#### GPU
```bash
export CUDA_VISIBLE_DEVICES=0
./run_kaggle.sh --use-gpu
```
sample log:
```
Reading processed data
Sparse fea = 26, Dense fea = 13
Defined train indices...
Randomized indices across days ...
Split data according to indices...
Reading processed data
Sparse fea = 26, Dense fea = 13
Defined test indices...
Randomized indices across days ...
Split data according to indices...
time/loss/accuracy (if enabled):
Finished training it 1024/306969 of epoch 0, 7.90 ms/it, loss 0.518843
Finished training it 2048/306969 of epoch 0, 7.90 ms/it, loss 0.504157
Finished training it 3072/306969 of epoch 0, 7.87 ms/it, loss 0.498123
Finished training it 4096/306969 of epoch 0, 7.89 ms/it, loss 0.487220
Finished training it 5120/306969 of epoch 0, 7.89 ms/it, loss 0.486305
Finished training it 6144/306969 of epoch 0, 7.92 ms/it, loss 0.481553
Finished training it 7168/306969 of epoch 0, 7.88 ms/it, loss 0.477103
Finished training it 8192/306969 of epoch 0, 7.86 ms/it, loss 0.476559
Finished training it 9216/306969 of epoch 0, 7.86 ms/it, loss 0.474173
Finished training it 10240/306969 of epoch 0, 7.91 ms/it, loss 0.471978
Testing at - 10240/306969 of epoch 0,
 accuracy 77.453 %, best 77.453 %
```
Tested GPU: NVIDIA TITAN RTX