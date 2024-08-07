[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10359141&assignment_repo_type=AssignmentRepo)
![DRAW](/public/draw-title.jpg)

# DRAW : Drawing Read Analysis Write

#### [웹 데모 보러가기](https://pdf-demo.jaewook.me)
#### [프로젝트 문서 보러가기](https://draw-docs.home.jaewook.me)

![DRAW poster](public/draw-poster.jpeg)

DRAW 프로젝트는 공사 도면 PDF 파일에서 도면 데이터 인식 및 데이터 추출을 하고, 추출한 데이터를 이용하여 도면 파일 위에 인식한 데이터를 제공하고 사용할 수 있게 합니다. 2023년 국민대학교 캡스톤 프로젝트로, 아이콘(AICON)과 산학협력 프로젝트로 진행하였습니다.

## 프로젝트 목표

저희 프로젝트의 목표는 건설 현장에서의 업무 효율성을 높이기 위해 OCR(광학 문자 인식) 기술과 Object Detection(물체 검출) 기술을 활용하여 건설 현장에서 사용되는 도면을 디지털화하고, 물체를 자동으로 인식하여 정보를 수집하고. 수집된 데이터를 WebAssembly와 canvas 기술을 이용해 사용자에게 유용한 정보를 새롭게 제공하거나 정보에 빠르게 접근할 수 있게 하는 것입니다. 이를 통해, 일일이 수작업으로 도면을 해석하고 정보를 수집하는 번거로운 작업을 줄이고, 건설 현장에서의 업무 효율성을 높일 수 있습니다.

## 시스텀 구성

![시스템 구성](/public/drawing.jpg)

## 수행 내용

### ML 파트

프로젝트 목표를 달성하기 위해 먼저 Object detection에서 많이 사용하는 Single-Stage Methods를 사용합니다. 이는 주로 원본 이미지를 고정된 사이즈의 그리드 영역으로 나누는데 이때 알고리즘은 각 영역에 대해 형태와 크기가 미리 결정된 객체의 고정 개수를 예측하게 됩니다. 해당 기술을 통해 도면 내에서 심볼이 어디에 위치하는지 위치정보를 찾고 해당 객체를 추출합니다. 이후 추출한 이미지 객체에서 텍스트를 추출하기 위해서 OCR을 활용합니다. OCR 또한 object detection과 유사한 과제입니다. 이는 해당 Pretrained Model을 가져와 우리의 과제에 적용시켜 성능을 얻고 이를 분석하게 됩니다. 최종적으로 도면 내에서 정보를 얻는 파이프 라인 구조를 구현하는 것이 목표입니다.

### FE 파트

저장된 데이터들은 분석을 통해 사용자에게 유용한 정보를 새롭게 제공할 수도 있을 것입니다. 이러한 목표를 달성하기 위해서 웹에서 WebAssembly와 canvas를 이용할 것입니다. 먼저, 사용자들이 도면, 사진과 같은 다양한 포맷의 파일을 웹 서비스에 업로드하고 별도의 플러그인이나 외부 프로그램 없이 해당 파일들을 웹 상에서 열람 가능하게 하는 자체 뷰어를 개발할 예정입니다. 이 뷰어는 대용량의 도면 데이터를 안정적이고 빠르게 처리하기 위해 C/C++로 개발된 모듈을 이용합니다. 해당 모듈은 WebAssembly를 통해 로드되고, JavaScript를 이용하는 것 보다 적은 메모리와 빠른 성능으로 실행됩니다. 다음으로 도면 정보를 담고 있는 PDF와 같은 파일에서 분석을 통해 얻어진 정보들로 뷰어의 개념을 확장하는 기능들을 개발할 예정입니다. 이러한 기능들에는 레이어 기능과 원본 데이터를 보여주면서 동시에 그 위에 추가적인 데이터를 보여주거나 드로잉, 링크, 편집 등의 기능들이 있습니다. 이를 개발하기 위해 canvas를 이용할 예정입니다. 궁극적으로는 뷰어가 아닌 에디터의 형태를 갖춘 실시간 협업이 가능한 모듈로 이 프로젝트를 발전시킬 것입니다. 이 모듈은 독립적으로 동작이 가능할 것이며, 건설에 활용되는 서비스의 각 부분에 탑재가 될 것입니다.


## 팀 소개

|    |    |
|:--:|:--:|
| <img width="300" src="/public/images/profile_jaewook.jpg" alt="Jaewook Ahn" /> | <img width="300" src="/public/images/profile_seungjin.jpg" alt="Seungjin Han" /> |
| 안재욱 (****1643) | 한승진 (****1512) |
| Frontend | Machine Learning |
| [Jaewoook](https://github.com/Jaewoook) | [SeungjinHan](https://github.com/seungjindes)
| dev.jaewook@gmail.com | gkstmdwls1999@gmail.com |
