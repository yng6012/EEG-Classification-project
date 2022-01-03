# ReadMe
## Default
### 폴더 설명
* brain_data8~50 : 학습할 데이터를 넣어두는 폴더
* brain_data8~50_t : 검증할 데이터를 넣어두는 폴더
* ckpt : 모델을 저장하는 폴더/ 일정 시간마다 모델을 저장 .\ckpt\high 폴더 내부에는 순위권의 accuracy와 낮은 loss를 .csv의 번호에 맞게 모델을 저장함.
* EEG-han2048-raw data : hanning window 2048 length, 150 step으로 계측한 STFT 데이터. 5000Hz의 계측속도로 6초간 데이터 측정을 진행. 8~30Hz의 데이터 보유
* gan_action_file : GAN 프로그램을 통해 증식된 action 데이터 파일
* gan_normal_file : GAN 프로그램을 통해 증식된 normal 데이터 파일
* han2048 : DataMining 프로그램으로 선별된 데이터 폴더

### 학습할 데이터는 다음의 순서를 따른다.
1. Raw data set을 DataMining program에 적용시켜 사용가능한 EEG 데이터 선별을 진행한다. (필수)
2. 선별된 데이터는 IOUGAN program을 통해 데이터 증식을 진행한다. (선택)
3. 증식된 데이터는 Pearson program을 통해 데이터 감소를 진행한다. (선택)
4. 위의 데이터 전처리가 모두 끝난 데이터를 일정 비율로 나눠 학습 폴더와 검증 폴더에 분배한다.
5. Main_LSTM program을 이용하여 학습을 진행한다.

### 필요 Python lib
* Python 3.6.13
* Numpy
* Pandas
* Matplotlib
* sklearn

## 프로그램 공통 변수
- n_times : 사용할 STFT 데이터의 시간축 개수를 입력한다. (준비된 데이터는 201개의 시간축 데이터를 가짐)
- n_frequency : 사용할 STFT 데이터의 주파수축 개수를 입력한다. (준비된 데이터는 79개의 주파수 데이터를 가짐)
- n_outputs : 분류되는 Class 개수를 나타낸다. (action , normal 로 2개)
- last_neurons, n_neurons : Stacked LSTM 구조의 각 층마다 Neuron의 개수를 정함. 
- limit_part : normalization할 단계 수를 나타냄. (normalization 사용안할 경우 무시)
- important_index : STFT 데이터에서 학습에 중요 시간영역을 따로 표시 [약 3.15~3.75초 사이]
- normal_range_index : STFT 데이터 전 후방 사용 불가능 데이터 제외한 normal data의 범위
- window : 학습에 사용할 시간 데이터의 크기. 단 important index의 차이보다 커야 함. ex) 30>(125-105) 의 경우에는 가능하지만, 15<(125-105)이기때문에 15로 설정할 경우 error 발생.
- step : important_index를 이용한 데이터 증식시 window stride의 step수.
- learning_rate : adam optimizer의 learning rate
- train_keep_prob, test_keep_prob : 학습시 drop out rate 설정. 1.12.0 은 보존되는 weight 비율을 설정하기 때문에 1에 가까울 수록 모두 살리겠다는 뜻을 가짐.
- batch_size : epoch당 학습할 데이터의 수, GPU 메모리 에러가 발생할 경우 값을 줄이는 것을 추천.

## Datamining_final
* Tensorflow 1.12.0
### 변수 설명
- want_round : 프로그램을 반복할 횟수를 정함. 0round 부터 시작하여 설정값의 +1만큼 진행.
- want_loop_percent : 검증에 반복 성공 빈도율을 나타냄. 1에 가까운 값일 수록 선별 난이도가 증가함.
- standard_convergence : 최소 round 반복 횟수 및 round의 accuracy와 loss 수렴으로 인한 종료 기준 epochs수. 값이 크면 클 수록, 종료되는 시간이 늦어지지만 학습 변동(안장점 탈출)을 기다릴 수 있음.
- data_base_path_array : Data Mining을 진행할 data set을 나타냄. 리스트 형식이며 train_file_path_array 리스트와 index match를 이용하여 Data Mining을 다수 dataset에 진행 가능함.
- train_file_path_array : Data Mining의 결과를 보여주는 폴더. data_base_path_array의 index data set에 따라 결과가 나타남.

### 프로그램 사용방법은 다음 순서를 따른다.
1. **data_base_path_array** 와 **train_file_path_array**에 입력한 폴더가 현재 path에 존재하는지 확인한다. 존재하지 않을 경우 폴더를 생성한다.
2. data_base_path_array의 데이터가 준비되었다면 위 변수 설명에 따라 변수를 확인 후, 프로그램을 실행한다.
3. 프로그램은 설정한 round만큼 반복 실행이 되며, 일정 loss와 accuracy가 반복될 경우 round는 종료된다.
4. 최종적으로 프로그램이 완료될 경우, train_file_path_array에 기입한 폴더로 들어가 **final_round**폴더 내부의 **success_data**의 파일을 사용한다. 해당 파일들이 선별된 STFT EEG 데이터이다.

## IOUGAN_final
* Tensorflow 2.1.0

### 변수 설명
- data_base_path : 증식할 데이터의 원본 데이터(Real data), 단 내부에는 하나의 Class에 대한 데이터만 들어있어야 함. ex) action과 normal중 하나만 해당 폴더 내부에 있어야함. 한 학습당 하나의 Class만 존재해야 함.
- gan_folder_path : 증식된 gan_data가 저장되는 폴더. 없을 시 생성해야 함.
- gan_data_name : 증식된 gan_data가 가지게 되는 이름. 뒤에는 숫자가 붙음. 하나의 gan 파일당 5개의 데이터를 가지게 됨.

### 프로그램 사용방법은 다음 순서를 따른다.
1. **data_base_path**에는 하나의 Class 폴더만 넣어두고 학습을 진행한다. (여러개의 Class 폴더를 넣고 학습을 진행할 경우, 혼합된 결과가 나오게 됨.)
2. 외의 변수는 공통 변수의 설명을 참고하여 설정한다.
3. GAN 데이터를 저장할 폴더의 경로를 **gan_folder_path**에 입력한다.
4. GAN 데이터 파일의 이름을 **gan_data_name**에 입력한다.(경로x)

## Pearson_final
* Tensorflow 1.12.0

### 변수 설명
- want_data : Pearson 프로그램을 반복할 횟수. 전체 데이터 수에 비례해서 많이 돌릴 수록 정확한 Pearson 상관계수 결과를 확인할 수 있다.
- key : 상관관계를 비교할 Class를 선택한다.
- standard_score : 상관관계 점수의 기준점수를 

### 프로그램 사용방법은 다음 순서를 따른다.
1. **key**변수를 이용하여 **data_base_path**폴더 내부의 Class를 하나 가져온다. 가져온 Class는 Pearson 상관관계 비교를 진행한다. **standard_score**는 피어슨 기준 점수이며 0.8 이상을 지정하는 것을 추천한다.
2. **want_data**를 이용하여 피어슨 상관관계 프로그램을 반복할 횟수를 정한다. Random한 2개의 데이터를 비교하는 프로그램이므로, **watn_data**변수는 크게 설정하는 것을 추천한다.
3. 프로그램이 모두 진행되면 **cost_data**와 **adr_cost_data**를 그래프를 통해 확인 및 비교함으로 써, 데이터간에 상관관계를 비교할 수 있다.
4. [1]번 그래프 프로그램을 이용하여 **adr_cost_data[frequency index]**의 **frequency index**를 이용해 시간에 따른 피어슨 점수 분포를 각 주파수 영역으로 확인할 수 잇다.
5. [2]번 그래프 프로그램은 **cost_data**를 이용하여 주파수 영역에 분포된 피어슨 점수를 확인할 수 있도록 해준다.
6. [3]번 그래프 프로그램을 이용하여 두개의 **cost_data**를 비교할 수 있게해준다. a와 b에 비교할 두개의 **cost_data**배열을 복사 및 붙여넣기 하면 확인할 수 있으며, 보통 action과 normal의 비교를 진행 할 때 사용된다.
## Main_LSTM_final
* Tensorflow 1.12.0

### 변수 설명
- max_load : 일정 시간마다 ckpt에 저장할 ckpt 파일의 최대 개수.
- high_max_load : 최대 accuracy 저장되는 ckpt 파일의 최대 개수.
- want_accuracy : 학습 강제 종료 accuracy 지표.
- last_accuracy : accuracy의 초기값. 0으로 설정한다.
- load_number : 저장된 ckpt 파일의 모델 번호를 기입할 경우 해당 모델을 불러와 검증 or 이어서 학습을 진행한다.
- n_epochs : 학습을 진행할 횟수.
- data_base_path : 학습(Train) 데이터를 모아두는 폴더 경로
- test_data_base_path : 검증(Test) 데이터를 모아두는 폴더 경로
- ckpt_base_path : 주기적 횟수 마다 학습된 ckpt 모델 파일 저장 경로
- high_weight_path_base : 학습 도중 가장 높은 accuracy의 ckpt 모델 파일 저장 경로

### 프로그램 사용방법은 다음 순서를 따른다.
1. 위에 전처리 프로그램에 의하여 최종적으로 나온 STFT EEG 데이터를 학습과 검증 데이터로 나눠 **data_base_path, test_data_base_path**경로에 넣어둔다.
2. 위의 공통 변수와 변수설명을 참고하여 변수값을 설정한다.
3. GAN 데이터를 사용하는 경우, GAN 데이터가 저장된 **action, normal**데이터 경로를 기입하여 [1] 프로그램을 실행한다.
4. 데이터의 normalization이 필요로 할 경우 [2] 프로그램을 실행한다.
5. 데이터의 limit part normalization이 필요한 경우 [3]프로그램을 실행한다. 단 [2]를 실행했을 경우 [3]은 실행하지 않는다.
6. 학습이 종료되고 **Accuracy, Loss, Max Accuracy**를 그래프로 확인하고자 한다면 [4]프로그램을 실행한다.