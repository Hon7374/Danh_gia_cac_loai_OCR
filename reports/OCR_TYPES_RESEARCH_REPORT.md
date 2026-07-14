# Bao cao nghien cuu rieng ve cac loai OCR, nguyen ly hoat dong va lien he voi demo

## 1. Mo dau

OCR la viet tat cua Optical Character Recognition, nghia la nhan dang ky tu quang hoc. Noi don gian, OCR la qua trinh bien noi dung chu trong anh, ban scan, anh chup dien thoai hoac file PDF dang hinh anh thanh van ban may tinh co the tim kiem, sao chep, luu tru va xu ly tu dong.

Vi du, khi co mot cong van giay duoc scan thanh PDF, may tinh chi nhin thay do la mot buc anh gom cac diem anh. Neu muon tim so ky hieu, ngay ban hanh, trich yeu, noi nhan hoac noi gui, ta can mot he thong OCR de doc lai cac dong chu trong anh. Sau do, cac ky thuat xu ly ngon ngu, regex hoac mo hinh hieu bo cuc tai lieu nhu LayoutLMv3 moi co the trich xuat thong tin co cau truc.

Trong demo hien tai, OCR khong chi la mot thu vien duy nhat. Demo duoc thiet ke nhu mot pipeline benchmark, nghia la cung mot tai lieu dau vao se duoc dua qua nhieu engine OCR khac nhau, sau do so sanh toc do, do chinh xac va chat luong van ban dau ra. Registry hien tai kich hoat bon engine: Tesseract, EasyOCR, PaddleOCR ket hop VietOCR va PaddleOCR-VL.

Bao cao nay khong chi trinh bay nhung OCR co trong demo, ma mo rong ra cac loai OCR pho bien trong thuc te: OCR truyen thong, OCR dua tren deep learning, OCR canh tu nhien, OCR tai lieu co bo cuc phuc tap, OCR viet tay, OCR da ngon ngu, OCR tren thiet bi bien, OCR cloud API va OCR ket hop mo hinh thi giac-ngon ngu.

## 2. OCR giai quyet bai toan gi?

Ve ban chat, OCR giai quyet ba cau hoi:

1. Trong anh co chu o dau?
2. Cac chu do la ky tu, tu, dong hay cau nao?
3. Ket qua van ban nen duoc sap xep va hieu theo thu tu nao?

Voi tai lieu don gian, chang han mot trang A4 nen trang chu den, OCR co the chi can phat hien dong chu va nhan dang tung dong. Voi tai lieu phuc tap hon, nhu hoa don, bang bieu, bieu mau hanh chinh, van ban co con dau, chu ky, tieu de hai cot, phu luc, footnote, OCR can them kha nang phan tich bo cuc. Neu khong hieu bo cuc, engine co the doc dung tung chu nhung ghep sai thu tu, dan den ket qua kho su dung.

Trong bai toan cong van tieng Viet cua demo, OCR gap cac kho khan dac thu:

- Tieng Viet co dau, nen sai dau co the lam thay doi nghia.
- Tai lieu hanh chinh co nhieu mau trinh bay: quoc hieu, tieu ngu, so ky hieu, dia danh-ngay thang, trich yeu, can cu, noi nhan.
- File dau vao co the la PDF sinh tu Word, PDF scan, anh chup, anh nghieng, nen xam, anh mo hoac co nen khong deu.
- Mot so trang co bang, dieu khoan, danh sach, phu luc, lam cho thu tu doc kho hon.
- OCR tot chua du; he thong con can hau xu ly de trich xuat truong thong tin dung.

## 3. Pipeline OCR hien dai

Mot he thong OCR hien dai thuong khong chi co buoc "doc chu". No la mot chuoi xu ly gom nhieu tang.

### 3.1. Dau vao

Dau vao co the la:

- Anh PNG, JPG, TIFF.
- PDF scan, trong do moi trang la anh.
- PDF co text layer san, tuc la van ban da nam trong file va co the trich xuat truc tiep.
- Anh chup tu dien thoai, co phoi canh, bong do, nhieu nhieu.
- Tai lieu da ngon ngu, vua co tieng Viet, tieng Anh, so, ky hieu, bang bieu.

Trong demo, file upload duoc chuyen thanh anh trang bang module xu ly PDF. Sau do moi trang anh se duoc chay qua OCR.

### 3.2. Tien xu ly anh

Tien xu ly co vai tro lam anh "de doc" hon cho OCR. Cac ky thuat thuong gap:

- Chuyen anh mau sang anh xam de giam do phuc tap.
- Khu nhieu de xoa hat scan, vet mo, nen ban.
- Can bang tuong phan de chu noi hon so voi nen.
- Cat vien trang thua.
- Sua nghieng trang, con goi la deskew.
- Nhi phan hoa, bien anh thanh den-trang.
- Xu ly nen khong deu, bong do hoac anh chup duoi anh sang kem.
- Lam net nhe, giup bien chu ro hon.

Trong demo, tien xu ly OpenCV duoc thiet ke kha can than. File `app/services/preprocess.py` co cac buoc:

- Doc anh bang OpenCV.
- Chuyen sang grayscale.
- Phan tich profile trang bang ty le diem trang, diem toi, diem trung gian va do tuong phan.
- Neu trang da sach, he thong cho di qua gan nhu nguyen ban de tranh lam hong anh tot.
- Neu trang can xu ly, he thong khu nhieu bang fastNlMeansDenoising.
- Sua nghieng bang minAreaRect.
- Tang tuong phan bang CLAHE.
- Neu anh thap tuong phan hoac nen khong deu, dung adaptive threshold.
- Neu anh tuong doi tot, dung unsharp mask nhe thay vi nhi phan hoa manh.

Diem quan trong la tien xu ly khong phai luc nao cung tot. Voi anh scan sach, nhi phan hoa qua manh co the lam mat dau tieng Viet, lam dut net chu, hoac lam cac dau cham nho bien mat. Vi vay demo co "quality gate" de anh sach duoc giu nguyen.

### 3.3. Phat hien vung chu

Buoc nay tra loi cau hoi: chu nam o dau trong anh?

Co nhieu cap do phat hien:

- Phat hien ky tu rieng le.
- Phat hien tu.
- Phat hien dong chu.
- Phat hien doan van.
- Phat hien block tai lieu nhu tieu de, bang, anh, chu ky, con dau.

OCR truyen thong thuong dua vao phan tich connected components, projection profile va quy tac hinh hoc. OCR hien dai thuong dung neural network, chang han CRAFT, DBNet, EAST, YOLO-like detector hoac cac text detector rieng.

Trong demo, Tesseract tu phan tich layout va tra box theo tu. EasyOCR phat hien vung chu roi nhan dang. PaddleOCR co thanh phan detection rieng, sau do recognition. PaddleOCR-VL co huong hieu tai lieu rong hon, co the sinh markdown/json va nhan dien layout.

### 3.4. Nhan dang chu

Sau khi co vung chu, engine can bien hinh anh thanh chuoi ky tu. Day la phan "recognition".

Cac cach nhan dang chinh:

- Template matching: so hinh ky tu voi mau co san.
- Feature engineering + classifier: trich dac trung nhu canh, goc, net doc, net ngang roi dung SVM, kNN, HMM.
- CNN + RNN + CTC: phu hop voi anh dong chu, khong can cat tung ky tu.
- Attention encoder-decoder: xem anh dong chu nhu mot chuoi can sinh ra tung ky tu.
- Transformer OCR: dung self-attention de hoc quan he khong gian va chuoi ky tu.
- Vision-language model: mo hinh lon co kha nang nhin anh va sinh text/markdown, khong chi doc chu don le.

Trong demo, VietOCR la vi du cua recognition toi uu cho tieng Viet. PaddleOCR co the detect va recognize, nhung khi bat refine, demo dung PaddleOCR de lay box, cat tung vung chu va dua qua VietOCR de nhan dang lai. Cach nay phu hop khi VietOCR duoc fine-tune tren du lieu cong van tieng Viet.

### 3.5. Hau xu ly

Ket qua OCR tho thuong con loi:

- Sai dau tieng Viet.
- Sai chu giong nhau: O/0, l/1/I, rn/m, S/5.
- Ghep sai dong.
- Mat dau cau.
- Thua khoang trang.
- Doc sai thu tu cot.
- Nham giua so ky hieu va ngay thang.

Hau xu ly giup sua hoac khai thac ket qua:

- Chuan hoa khoang trang, xuong dong, Unicode.
- Tu dien va language model de sua loi chinh ta.
- Regex de bat mau ngay thang, so ky hieu, ma so.
- Rule-based extraction de lay cac truong quen thuoc.
- Layout-aware extraction bang LayoutLM, LayoutLMv2, LayoutLMv3, Donut, DocTR, Nougat, Kosmos, LLaVA-like document models.

Trong demo, sau OCR co buoc tinh chat luong, tinh CER/WER neu co ground truth, va trich xuat truong cong van bang LayoutLMv3 neu co model fine-tuned. Neu chua co model hoac model khong san sang, he thong fallback sang rule/regex.

### 3.6. Danh gia

Danh gia OCR khong nen chi nhin bang mat. Cac chi so thuong dung:

- CER, Character Error Rate: ty le loi theo ky tu.
- WER, Word Error Rate: ty le loi theo tu.
- Precision/Recall/F1 cho phat hien box chu.
- Exact match cho truong thong tin.
- Thoi gian xu ly.
- Tai nguyen can dung: CPU, GPU, RAM, dung luong model.
- Do on dinh khi anh xau, anh nghieng, scan mo.

Trong demo, `app/services/metrics.py` tinh CER va WER dua tren edit distance. Text truoc khi so sanh duoc chuan hoa ve chu thuong va rut gon khoang trang. Ket qua benchmark luu trong CSV gom engine, variant, status, thoi gian, CER, WER, loi va preview text.

## 4. Phan loai OCR theo cong nghe

### 4.1. OCR truyen thong dua tren quy tac va dac trung thu cong

Day la lop OCR ra doi som. He thong thuong xu ly anh de tach ky tu, sau do so sanh voi mau hoac dung dac trung hinh hoc.

Quy trinh thuong la:

1. Nhi phan hoa anh.
2. Tach dong, tach tu, tach ky tu.
3. Trich dac trung cua ky tu.
4. So sanh voi tap mau hoac phan loai bang mo hinh nhe.
5. Ghep ky tu thanh tu va cau.

Uu diem:

- Nhe, chay nhanh tren CPU.
- Giai thich duoc, de kiem soat bang quy tac.
- Tot voi font in ro, nen sach, bo cuc don gian.

Nhuoc diem:

- Yeu voi anh chup ngoai doi, font la, anh nghieng, nen phuc tap.
- Kho xu ly chu viet tay.
- Kho xu ly ngon ngu co dau phuc tap neu du lieu huan luyen han che.
- Viec cat ky tu co the sai khi chu dinh, dau tieng Viet nho, hoac scan kem.

Tesseract ban dau thuoc nhom truyen thong, nhung cac phien ban moi da tich hop LSTM nen khong con la OCR thuan quy tac.

### 4.2. OCR dua tren LSTM/sequence model

OCR bang LSTM coi dong chu nhu mot chuoi. Thay vi cat tung ky tu, mo hinh nhin anh dong chu va du doan chuoi ky tu. Dieu nay giam phu thuoc vao buoc segmentation ky tu.

Kien truc pho bien:

- CNN trich dac trung anh.
- RNN/LSTM/BiLSTM doc dac trung theo chieu ngang.
- CTC loss canh chinh dau ra ky tu voi anh ma khong can biet vi tri tung ky tu.

Uu diem:

- Tot hon OCR truyen thong voi dong chu lien mach.
- Khong can cat tung ky tu.
- Phu hop voi van ban in, scan, anh dong chu.

Nhuoc diem:

- Van co the yeu voi layout phuc tap.
- Can du lieu huan luyen theo ngon ngu.
- Viec nhan dang dau tieng Viet phu thuoc chat luong tap train.

Tesseract 4+ la vi du noi bat cua OCR co LSTM. Trong demo, Tesseract duoc goi qua pytesseract, dung language data `vie` va `eng` neu co.

### 4.3. OCR dua tren CNN-RNN-CTC

Day la huong rat pho bien trong cac OCR deep learning truoc Transformer. Mo hinh thuong co CNN de trich dac trung hinh anh, RNN de doc chuoi, CTC de sinh text.

Uu diem:

- Doc dong chu tot.
- Toc do kha nhanh.
- Khong can label vi tri tung ky tu.
- Hoat dong tot voi text in va nhieu font.

Nhuoc diem:

- Kho voi dong qua dai hoac bo cuc phuc tap.
- Thu tu doc phu thuoc vao detector va cach cat crop.
- It "hieu" noi dung tai lieu, chu yeu nhan dang ky tu.

EasyOCR va nhieu pipeline OCR open-source su dung tu tuong gan voi nhom nay, ket hop detector va recognizer deep learning.

### 4.4. OCR dua tren Attention va Transformer

Transformer giup mo hinh nhin chuoi ky tu linh hoat hon. Thay vi doc tu trai sang phai bang RNN, attention cho phep mo hinh tap trung vao cac vung anh lien quan khi sinh tung ky tu.

Kien truc thuong gap:

- CNN hoac Vision Transformer lam encoder anh.
- Transformer decoder sinh chuoi ky tu.
- Attention canh chinh giua token dau ra va dac trung anh.

Uu diem:

- Manh voi chuoi phuc tap.
- Hoc duoc quan he dai hon.
- De ket hop voi pretraining va fine-tuning.
- Phu hop voi ngon ngu co dau neu du lieu tot.

Nhuoc diem:

- Can nhieu tai nguyen hon.
- Co the cham tren CPU.
- Neu mo hinh sinh text tu do, co nguy co sua sai theo "doan" thay vi doc dung tung ky tu.

VietOCR trong demo la vi du gan voi nhom OCR recognition hien dai cho tieng Viet. Demo co model `models/vietocr-congvan/transformerocr.pth`, tuc la co huong fine-tune cho mien du lieu cong van.

### 4.5. OCR end-to-end

OCR end-to-end co gang giai quyet detection va recognition trong mot he thong thong nhat. Thay vi tach thanh nhieu module doc lap, mo hinh co the hoc truc tiep tu anh tai lieu sang text hoac sang cau truc.

Uu diem:

- Giam loi lan truyen giua detector va recognizer.
- Co the toi uu tong the.
- Phu hop voi bai toan can output co cau truc.

Nhuoc diem:

- Can du lieu huan luyen lon.
- Kho debug tung buoc.
- Neu sai, kho biet sai do phat hien vung chu hay do nhan dang.

Mot so mo hinh document AI moi nhu Donut, Nougat, TrOCR, DocTR, PaddleOCR-VL, GLM-OCR va cac vision-language model co xu huong tien gan den OCR end-to-end hoac OCR ket hop hieu tai lieu.

### 4.6. OCR layout-aware

OCR layout-aware khong chi doc chu, ma con quan tam vi tri va vai tro cua chu trong trang. No co the phan biet:

- Tieu de.
- Doan van.
- Bang.
- Header/footer.
- Chu thich.
- So trang.
- Con dau, chu ky, logo.
- Truong thong tin trong bieu mau.

Huong nay rat quan trong voi cong van, hoa don, hop dong, phieu khai, ho so hanh chinh.

Cong nghe thuong gap:

- Layout detection.
- Table structure recognition.
- Key-value extraction.
- Token classification theo vi tri.
- Vision-language model sinh markdown/json.

Trong demo, PaddleOCR-VL va LayoutLMv3 lien quan nhieu den nhom nay. PaddleOCR-VL co the tra text kem cau truc layout. LayoutLMv3 dung token, box va anh trang de trich xuat truong nhu so ky hieu, ngay ban hanh, trich yeu.

### 4.7. OCR viet tay

OCR viet tay, hay handwriting recognition, kho hon OCR in may vi chu moi nguoi khac nhau, net chu co the noi lien, nghieng, thieu dau, khoang cach khong deu.

Co hai dang:

- Offline handwriting recognition: nhan dang tu anh chu viet tay.
- Online handwriting recognition: nhan dang tu net ve theo thoi gian, co toa do but, toc do, thu tu net.

Uu diem cua online la co them thong tin cach nguoi viet tao net, nen de hon. Offline chi co anh cuoi cung nen kho hon.

Trong bai toan cong van demo, OCR viet tay khong phai muc tieu chinh. Tuy nhien chu ky, ghi chu tay, but phe hoac dau moc co the xuat hien trong van ban hanh chinh. Cac engine OCR thong thuong thuong khong doc tot chu ky hoac chu viet tay.

### 4.8. OCR canh tu nhien

OCR canh tu nhien doc chu trong anh doi thuc, vi du bien bao, bang ten duong, man hinh, hoa don chup bang dien thoai, anh san pham.

Khac voi scan A4, canh tu nhien co:

- Phoi canh nghieng.
- Anh sang khong deu.
- Nen phuc tap.
- Chu cong, chu bien dang.
- Vat the che khuat.
- Nhieu font va mau sac.

EasyOCR va PaddleOCR thuong duoc dung tot cho nhom nay vi co detector manh. Tesseract co the kem hon neu anh khong duoc tien xu ly ky.

### 4.9. OCR da ngon ngu

OCR da ngon ngu can nhan dang nhieu bang chu va ngon ngu, vi du tieng Viet, Anh, Trung, Nhat, Han, Thai, Arab. Thach thuc la moi ngon ngu co bo ky tu, dau, cach viet va quy tac tach tu khac nhau.

Voi tieng Viet, diem can chu y:

- Dau thanh va dau nguyen am nho, de mat khi anh mo.
- Unicode co the bi loi neu khong chuan hoa.
- Mot tu sai dau van "trong co ve dung" nhung sai nghia.
- Tu viet tat hanh chinh nhu QD, TT, ND, CP, UBND, NHNN, BGDĐT can duoc giu dung.

Trong demo, Tesseract dung `vie` va `eng`; PaddleOCR chay `lang="vi"`; VietOCR duoc fine-tune theo cong van tieng Viet; dieu nay phu hop voi muc tieu tai lieu hanh chinh Viet Nam.

### 4.10. OCR cloud API

OCR cloud API la dich vu OCR chay tren may chu cua nha cung cap. Nguoi dung gui anh/PDF len API va nhan ket qua text/json.

Vi du nhom nay:

- Google Cloud Vision OCR.
- Azure AI Document Intelligence.
- Amazon Textract.
- ABBYY Cloud OCR.
- Z.ai/GLM-OCR endpoint.
- Cac API document parsing khac.

Uu diem:

- Khong can cai model nang tren may local.
- Thuong co model manh va cap nhat lien tuc.
- Co kha nang layout, table, key-value extraction.
- De tich hop vao he thong web.

Nhuoc diem:

- Can Internet va API key.
- Ton chi phi theo so trang/so request.
- Du lieu nhay cam phai gui ra ngoai, can quan tam bao mat.
- Ket qua phu thuoc nha cung cap.

Trong demo, file `glm_ocr_engine.py` cho thay GLM-OCR co the chay bang API key hoac local HuggingFace neu cau hinh cho phep. README de cap endpoint layout parsing va bien moi truong `ZAI_API_KEY` hoac `GLM_OCR_API_KEY`.

### 4.11. OCR tren thiet bi bien

OCR tren thiet bi bien la OCR chay truc tiep tren dien thoai, may scan, kiosk, camera cong nghiep hoac may tinh khong co GPU manh.

Yeu cau:

- Model nhe.
- Do tre thap.
- Co the chay offline.
- Toi uu bang ONNX, TensorRT, OpenVINO, CoreML, TFLite.

Uu diem:

- Bao mat tot hon vi du lieu khong roi khoi thiet bi.
- Phan hoi nhanh.
- Khong phu thuoc Internet.

Nhuoc diem:

- Do chinh xac co the thap hon model lon.
- Kho cap nhat, kho fine-tune tai cho.
- Gioi han RAM/CPU.

Voi demo hien tai, huong local/offline cua Tesseract, EasyOCR, PaddleOCR va VietOCR co the xem la gan voi trien khai on-premise/local. Tuy nhien neu can chay tren mobile thuc su, can toi uu model nhe hon.

## 5. Cac OCR engine pho bien

### 5.1. Tesseract OCR

Tesseract la engine OCR ma nguon mo lau doi va pho bien. No ho tro nhieu ngon ngu, co language data rieng va co the chay offline. Tesseract phu hop voi tai lieu scan ro, font in thong dung, bo cuc khong qua phuc tap.

Nguyen ly tong quat:

- Tien xu ly anh va phan tich bo cuc.
- Tim dong, tu, ky tu hoac thanh phan chu.
- Nhan dang bang mo hinh LSTM trong cac phien ban moi.
- Tra text va thong tin box/confidence.

Uu diem:

- Mien phi, nguon mo, chay offline.
- Nhe hon nhieu model deep learning moi.
- De tich hop qua pytesseract.
- Co du lieu ngon ngu `vie` cho tieng Viet.
- Phu hop lam baseline trong nghien cuu.

Nhuoc diem:

- Can cai binary Tesseract ngoai Python.
- Nhan dang tieng Viet co dau khong phai luc nao cung tot.
- Yeu voi anh nghieng, nen phuc tap, anh chup tu dien thoai.
- Layout phuc tap co the lam sai thu tu doc.

Trong demo:

- Engine nam o `app/ocr_engines/tesseract_engine.py`.
- Demo tu tim binary Tesseract trong PATH, thu muc cai dat pho bien tren Windows/Linux/macOS va thu muc tessdata cua project.
- Dung pytesseract `image_to_data` de lay text, confidence va bounding box.
- Neu yeu cau `vie` nhung may thieu data, engine co fallback sang `eng` de demo khong bi dung hoan toan.
- Ket qua text duoc ghep theo tu, box tra ve theo tung tu.

Tesseract nen duoc xem la baseline: de chay, de so sanh, nhung khong phai lua chon manh nhat cho moi tai lieu cong van scan xau.

### 5.2. EasyOCR

EasyOCR la thu vien OCR deep learning de dung, ho tro nhieu ngon ngu va thuong duoc dung cho ca anh scan lan anh canh tu nhien.

Nguyen ly tong quat:

- Detector phat hien vung chu trong anh.
- Recognizer nhan dang noi dung vung chu.
- Tra danh sach box, text va confidence.

Uu diem:

- Cai dat va su dung tuong doi don gian.
- Ho tro nhieu ngon ngu, trong do co tieng Viet.
- Hoat dong tot hon Tesseract trong mot so anh chup/canh tu nhien.
- Tra box va confidence tien loi cho benchmark.

Nhuoc diem:

- Goi cai dat nang hon Tesseract vi phu thuoc PyTorch.
- Lan dau chay co the tai model.
- Tren CPU co the cham.
- Van co the sai dau tieng Viet hoac doc sai thu tu voi layout phuc tap.

Trong demo:

- Engine nam o `app/ocr_engines/easyocr_engine.py`.
- Demo chay EasyOCR trong subprocess worker rieng de tranh xung dot bo nho/model va giup timeout ro rang.
- Worker co timeout 30 phut.
- Neu chua cai package, engine bao skipped/error tuy tinh huong de dashboard van hien thi duoc.

EasyOCR phu hop lam engine deep learning tong quat trong benchmark, dac biet khi can so voi Tesseract va PaddleOCR.

### 5.3. PaddleOCR

PaddleOCR la bo cong cu OCR manh cua he sinh thai PaddlePaddle. No co nhieu pipeline: text detection, text recognition, angle classification, table recognition, document structure, PP-OCR series va cac ban da ngon ngu.

Uu diem:

- Do chinh xac cao voi nhieu loai tai lieu.
- Co detector va recognizer manh.
- Ho tro nhieu ngon ngu.
- Co ecosystem document AI rong hon OCR thuong.
- Co the chay CPU/GPU tuy cau hinh.

Nhuoc diem:

- Cai dat nang, de gap xung dot package tren Windows.
- Phu thuoc PaddlePaddle.
- Model va API co the thay doi theo version.
- Can cau hinh device va version can than.

Trong demo:

- Engine ket hop nam o `app/ocr_engines/paddle_vietocr_engine.py`.
- PaddleOCR duoc khoi tao voi `lang="vi"`.
- Demo tat mot so thanh phan nhu orientation classify, doc unwarping, textline orientation de giam do phuc tap khi chay local.
- Ket qua PaddleOCR duoc chuan hoa ve danh sach box gom text, confidence, bbox.
- Engine chay trong subprocess worker rieng, timeout 30 phut.

PaddleOCR la ung vien manh cho OCR tieng Viet in may, dac biet khi anh dau vao da duoc tien xu ly tot.

### 5.4. VietOCR

VietOCR la OCR recognition tap trung vao tieng Viet. Diem manh cua VietOCR la nhan dang dong chu tieng Viet co dau, dac biet khi duoc fine-tune tren dung mien du lieu.

Can phan biet:

- VietOCR khong nhat thiet la pipeline detection day du nhu PaddleOCR.
- No manh o recognition tren crop dong chu/tu/vung chu.
- Vi vay thuong can mot detector khac cat vung chu truoc.

Trong demo:

- VietOCR duoc dung nhu buoc refine cho PaddleOCR.
- PaddleOCR phat hien vung chu va lay box.
- He thong crop tung box tu anh goc.
- VietOCR nhan dang lai noi dung trong tung crop.
- Neu co model fine-tuned `models/vietocr-congvan/transformerocr.pth`, ket qua co the phu hop hon voi cong van tieng Viet.
- Vi VietOCR co the cham tren CPU, demo co bien cau hinh bat/tat refine.

Uu diem:

- Phu hop tieng Viet.
- Co the fine-tune theo bo du lieu cong van.
- Cai thien loi dau tieng Viet neu train tot.

Nhuoc diem:

- Can detector di kem.
- Cham neu refine tung crop tren CPU.
- Ket qua phu thuoc chat luong box dau vao.
- Neu box cat thieu dau hoac cat dinh dong, VietOCR van sai.

VietOCR trong demo la diem quan trong vi de tai huong den van ban hanh chinh tieng Viet, khong chi OCR tieng Anh tong quat.

### 5.5. PaddleOCR-VL

PaddleOCR-VL la huong OCR/document parsing moi hon, ket hop thi giac va ngon ngu de hieu tai lieu phuc tap. Khac voi OCR truyen thong chi tra text thuong, PaddleOCR-VL co the tra markdown, json, thong tin layout hoac noi dung co cau truc tuy cach chay.

Uu diem:

- Phu hop tai lieu co layout phuc tap.
- Co kha nang hieu bang, block, thanh phan tai lieu tot hon OCR dong chu thuan.
- Ket qua co the gan voi markdown/json, tien cho hau xu ly.

Nhuoc diem:

- Model nang.
- Cai dat va chay tren Windows CPU co the kho.
- Can version PaddleOCR/PaddlePaddle phu hop.
- Thoi gian chay dai hon engine nhe.

Trong demo:

- Engine nam o `app/ocr_engines/paddleocr_vl_engine.py`.
- Co hai cach chay: qua command `PADDLEOCR_VL_CMD` hoac import Python API `from paddleocr import PaddleOCRVL`.
- Engine doc cac file output sinh ra nhu `.md`, `.txt`, `.json`.
- Engine co ham trich text de lay noi dung tu cau truc phuc tap.
- Cung chay qua worker rieng va co timeout.

PaddleOCR-VL phu hop de mo rong demo tu "doc chu" sang "hieu tai lieu".

### 5.6. GLM-OCR va OCR bang vision-language model

GLM-OCR dai dien cho huong dung mo hinh thi giac-ngon ngu lon de doc va phan tich tai lieu. Mo hinh khong chi nhan dang ky tu, ma co the sinh markdown, layout text hoac noi dung co cau truc.

Uu diem:

- Co the hieu bo cuc va ngu canh tot hon OCR co dien.
- Phu hop tai lieu phuc tap.
- Co the tra ket qua gan voi markdown/json.
- Khi chay qua API, nguoi dung khong can cai model nang.

Nhuoc diem:

- Neu la API, can key, Internet, chi phi va can quan tam bao mat.
- Neu chay local, can GPU/RAM manh.
- Co nguy co hallucination: mo hinh sinh ra chu co ve hop ly nhung khong dung 100% voi anh.
- Danh gia can nghiem ngat bang ground truth.

Trong repo:

- Co file `app/ocr_engines/glm_ocr_engine.py`.
- Engine co the goi API endpoint voi `GLM_OCR_API_KEY` hoac `ZAI_API_KEY`.
- Neu khong co API key, code co huong chay local HuggingFace model `zai-org/GLM-OCR` neu cau hinh cho phep.
- Adapter `glm_ocr` duoc giu lai de tham khao du lieu cu nhung khong con dang ky trong demo.

### 5.7. Google Cloud Vision OCR

Google Cloud Vision OCR la dich vu OCR cloud manh, ho tro nhieu ngon ngu va anh canh tu nhien. No phu hop khi can ket qua nhanh ma khong muon cai dat model local.

Uu diem:

- Chat luong tot tren nhieu loai anh.
- Ho tro nhieu ngon ngu.
- Co API on dinh.
- Phu hop tich hop backend.

Nhuoc diem:

- Ton phi.
- Can gui du lieu len cloud.
- Khong phai luc nao toi uu rieng cho cong van Viet Nam.

### 5.8. Azure AI Document Intelligence

Azure AI Document Intelligence, truoc day thuong duoc biet den voi ten Form Recognizer, la dich vu document AI. No khong chi OCR ma con trich xuat key-value, bang, form va co model tuy bien.

Uu diem:

- Manh voi hoa don, bieu mau, tai lieu doanh nghiep.
- Co table/key-value extraction.
- Co kha nang huan luyen model rieng.

Nhuoc diem:

- Chi phi va cloud dependency.
- Can cau hinh, quan ly resource.
- Du lieu nhay cam can xem xet chinh sach bao mat.

### 5.9. Amazon Textract

Amazon Textract la dich vu OCR/document extraction cua AWS. No tap trung vao text, form va table.

Uu diem:

- Tich hop tot voi AWS.
- Manh voi form va bang.
- Co output cau truc.

Nhuoc diem:

- Chi phi.
- Phu thuoc cloud.
- Can xu ly ngon ngu va format rieng neu tai lieu tieng Viet hanh chinh.

### 5.10. ABBYY FineReader/ABBYY OCR

ABBYY la giai phap OCR thuong mai lau doi, noi tieng ve chat luong OCR tai lieu scan.

Uu diem:

- Chat luong cao voi tai lieu scan.
- Ho tro nhieu ngon ngu.
- Co san pham desktop va enterprise.

Nhuoc diem:

- Thuong la phan mem/dich vu co phi.
- Do tuy bien va tich hop tuy goi san pham.
- Khong phai nguon mo.

### 5.11. TrOCR

TrOCR la huong OCR dua tren Transformer encoder-decoder. No co the dung cho printed text va handwritten text tuy model.

Uu diem:

- Kien truc hien dai.
- Co the fine-tune cho bai toan rieng.
- Phu hop nghien cuu deep learning OCR.

Nhuoc diem:

- Can du lieu va tai nguyen tinh toan.
- Khong phai pipeline document OCR day du neu khong co detector/layout.

### 5.12. Donut va OCR-free document understanding

Donut la huong "OCR-free", nghia la mo hinh nhin anh tai lieu va sinh truc tiep chuoi ket qua co cau truc ma khong can OCR trung gian truyen thong.

Uu diem:

- Co the sinh JSON/truong thong tin truc tiep.
- Tot cho bai toan hieu tai lieu neu duoc fine-tune.
- Giam phu thuoc vao OCR text trung gian.

Nhuoc diem:

- Can du lieu gan nhan cau truc.
- Kho debug.
- Co nguy co sinh sai neu khong kiem soat.

Huong nay co lien quan voi muc tieu cuoi cua demo: khong chi lay text, ma trich xuat thong tin cong van.

### 5.13. LayoutLMv3

LayoutLMv3 khong phai OCR engine theo nghia doc anh thanh text. No la mo hinh hieu tai lieu, dung text OCR, toa do box va anh trang de lam cac tac vu nhu token classification, key information extraction, document classification.

Trong demo:

- OCR engine tao text va boxes.
- LayoutLMv3 hoac fallback rule dung ket qua do de trich xuat truong.
- Cac truong muc tieu gom so ky hieu, ngay ban hanh, trich yeu, co quan ban hanh, noi gui, noi nhan, loai van ban.

Vay LayoutLMv3 nen duoc xem la tang sau OCR, khong thay the hoan toan OCR. Neu OCR sai, LayoutLMv3 cung bi anh huong. Neu OCR dung nhung box/thu tu sai, model layout cung co the trich xuat sai.

## 6. So sanh cac nhom OCR

| Nhom OCR | Diem manh | Diem yeu | Phu hop voi |
|---|---|---|---|
| OCR truyen thong | Nhe, nhanh, de chay offline | Yeu voi anh xau/layout phuc tap | Scan sach, baseline |
| LSTM/CTC | Tot voi dong chu in | Can du lieu ngon ngu, layout van kho | Tai lieu in, PDF scan |
| CNN/RNN deep OCR | Can bang giua toc do va do chinh xac | Cai dat nang hon, can model | Anh scan va anh chup |
| Transformer OCR | Manh voi chuoi va ngon ngu | Nang, can fine-tune | Tieng Viet, bai toan chuyen biet |
| Layout-aware OCR | Hieu bo cuc, bang, block | Nang, phuc tap | Cong van, hoa don, form |
| Vision-language OCR | Hieu ngu canh tot, sinh markdown/json | Co nguy co hallucination, can GPU/API | Tai lieu phuc tap |
| Cloud OCR | De tich hop, khong can cai model | Chi phi, bao mat, Internet | He thong doanh nghiep |
| Edge OCR | Offline, bao mat, nhanh tai thiet bi | Model nhe hon, do chinh xac co the kem | Mobile, kiosk, may scan |

## 7. Pipeline OCR trong demo

Demo hien tai co the mo ta bang luong sau:

1. Nguoi dung upload file anh/PDF.
2. Neu la PDF, he thong render moi trang thanh anh.
3. Moi trang anh duoc tao hai bien the: raw va opencv_preprocessed.
4. OpenCV tien xu ly anh neu can: grayscale, denoise, deskew, CLAHE, adaptive threshold hoac unsharp mask.
5. Cac OCR engine duoc chay tren tung trang/bien the.
6. Ket qua moi engine gom text, boxes, elapsed time, status, error, raw metadata.
7. Neu co ground truth, he thong tinh CER va WER.
8. He thong phan tich chat luong OCR va chon ket qua tot theo diem tong hop.
9. Ket qua OCR duoc dua sang buoc trich xuat truong bang LayoutLMv3 hoac rule-based fallback.
10. Dashboard hien thi so sanh engine, bieu do, bang ket qua va file CSV/JSON.

Cac engine trong registry hien tai:

- `tesseract`: baseline local/offline, goi qua pytesseract.
- `easyocr`: deep learning OCR, chay worker rieng.
- `paddle_vietocr`: PaddleOCR detect/recognize, tuy chon refine bang VietOCR.
- `paddleocr_vl`: document OCR/layout parsing hien dai, chay API Python hoac CLI adapter.

Adapter legacy khong kich hoat:

- `glm_ocr`: source duoc giu de doc ket qua cu, khong nam trong registry hien tai.

Tang sau OCR:

- `LayoutLMv3`: trich xuat truong cong van neu co model fine-tuned.
- Rule/regex fallback: dung khi LayoutLMv3 chua san sang hoac de on dinh MVP.

## 8. Vi sao demo chay nhieu OCR engine?

Khong co engine OCR nao tot nhat cho moi truong hop. Cung mot tai lieu, engine A co the doc dung dau tieng Viet nhung cham, engine B nhanh nhung sai layout, engine C manh voi scan sach nhung kem voi anh chup.

Chay nhieu engine giup:

- So sanh khach quan bang CER/WER.
- Chon engine tot nhat theo tung loai tai lieu.
- Phat hien engine nao bi loi cai dat/API.
- Danh gia anh huong cua tien xu ly OpenCV.
- Co co so khoa hoc cho bao cao NCKH.
- Ket hop dau ra de trich xuat truong on dinh hon.

Trong `layoutlmv3_postprocess.py`, demo con co y tuong uu tien nguon khac nhau khi hop nhat truong. Vi du EasyOCR, PaddleOCR-VL, Tesseract va Paddle-VietOCR co diem uu tien rieng tuy ngu canh. Dieu nay phan anh thuc te: output cuoi cung co the tot hon neu biet chon dung engine cho dung truong.

## 9. Cac loi OCR thuong gap va cach khac phuc

### 9.1. Sai dau tieng Viet

Vi du:

- "Quyet dinh" thay vi "Quyet dinh" co dau day du.
- "Ngan hang nha nuoc" thay vi "Ngan hang Nha nuoc".
- "bo sung" bi doc thanh "b6 sung" hoac "bô sung".

Nguyen nhan:

- Anh mo lam mat dau.
- Model khong duoc train tot cho tieng Viet.
- Tien xu ly lam bien mat dau nho.

Khac phuc:

- Dung model tieng Viet nhu VietOCR fine-tuned.
- Giu anh raw neu anh da sach.
- Tang DPI khi render PDF.
- Dung tu dien/chinh ta/hau xu ly theo mien cong van.

### 9.2. Nham ky tu giong nhau

Vi du:

- O va 0.
- I, l va 1.
- S va 5.
- Đ, D, O voi con dau scan mo.

Khac phuc:

- Hau xu ly theo mau. Neu dang ngay thang thi uu tien so; neu dang ten co quan thi uu tien chu.
- Dung regex cho so ky hieu, ngay thang.
- Fine-tune tren du lieu that.

### 9.3. Sai thu tu dong

Tai lieu co hai cot, bang bieu, header/footer co the lam OCR doc sai thu tu.

Khac phuc:

- Dung layout-aware OCR.
- Sap xep box theo toa do y/x co dieu kien.
- Nhan dien block truoc khi ghep text.
- Dung PaddleOCR-VL hoac LayoutLMv3 de hieu bo cuc.

### 9.4. Mat dong hoac thua dong

Nguyen nhan:

- Detector bo sot vung chu nho.
- Anh qua sang/toi.
- Threshold qua manh.
- Crop cat mat dau dong.

Khac phuc:

- Thu raw va preprocessed song song.
- Dieu chinh adaptive threshold.
- Tang DPI render PDF.
- Dung engine khac de doi chieu.

### 9.5. OCR dung text nhung trich xuat truong sai

Day la loi sau OCR. Vi du OCR doc dung ca trang, nhung regex lay nham ngay trong phan noi dung thay vi ngay ban hanh.

Khac phuc:

- Dung layout: ngay ban hanh thuong nam gan dau trang, ben phai.
- Dung rule theo cau truc cong van.
- Fine-tune LayoutLMv3 cho token classification.
- Ket hop nhieu ung vien va cham diem theo ngu canh.

## 10. Lien he voi de tai nghien cuu cong van

Voi de tai "Nghien cuu OCR, nguyen ly OCR, pipeline OCR hien dai, cac cong cu pho bien: Tesseract, PaddleOCR, EasyOCR", demo co y nghia nhu mot minh chung thuc nghiem:

- Tesseract dai dien cho OCR open-source lau doi, local/offline, lam baseline.
- EasyOCR dai dien cho OCR deep learning de dung, ho tro da ngon ngu.
- PaddleOCR dai dien cho pipeline OCR hien dai co detector/recognizer manh.
- VietOCR dai dien cho huong chuyen biet hoa tieng Viet va fine-tune theo mien cong van.
- PaddleOCR-VL dai dien cho xu huong document AI/layout-aware OCR.
- LayoutLMv3 dai dien cho tang hieu tai lieu sau OCR, giup trich xuat truong thong tin.
- OpenCV preprocessing dai dien cho tang xu ly anh dau vao, anh huong truc tiep den chat luong OCR.
- CER/WER dai dien cho cach danh gia dinh luong, tranh nhan xet cam tinh.

Noi cach khac, demo khong chi "goi OCR" ma the hien mot pipeline gan voi he thong thuc te:

Tai lieu dau vao -> tien xu ly -> OCR nhieu engine -> danh gia -> chon ket qua -> hau xu ly/trich xuat truong -> dashboard bao cao.

## 11. De xuat huong cai tien

### 11.1. Cai tien du lieu

- Xay dung tap ground truth lon hon cho cong van tieng Viet.
- Tach ground truth theo cap trang, dong, truong thong tin.
- Bao gom nhieu loai tai lieu: quyet dinh, thong tu, cong van, nghi dinh, thong bao.
- Bao gom scan sach, scan mo, anh nghieng, PDF sinh tu Word.

### 11.2. Cai tien OCR

- Fine-tune VietOCR tren crop dong chu cong van.
- Thu PaddleOCR voi cac cau hinh detection/recognition khac nhau.
- Them ensemble: chon ket qua theo truong, khong chi theo toan van.
- Them cloud OCR de lam benchmark tham chieu neu duoc phep gui du lieu.

### 11.3. Cai tien layout

- Gan nhan box cho cac truong cong van.
- Fine-tune LayoutLMv3 bang du lieu that.
- Them nhan dien bang va phu luc.
- Sap xep text theo block de giam sai thu tu doc.

### 11.4. Cai tien danh gia

- Ngoai CER/WER, them field-level accuracy.
- Do F1 cho tung truong: so ky hieu, ngay ban hanh, trich yeu.
- Do thoi gian tren CPU/GPU rieng.
- Do ty le engine skipped/error de danh gia kha nang trien khai.

### 11.5. Cai tien trien khai

- Tao che do "nhanh" chi chay Tesseract/Paddle.
- Tao che do "day du" chay tat ca engine.
- Cache ket qua model de lan chay sau nhanh hon.
- Cho phep chon trang dau, tat ca trang hoac gioi han so trang khi benchmark.
- Dong goi Docker full OCR rieng cho moi truong bao ve.

## 12. Ket luan

OCR la mot linh vuc rong, tu cac engine truyen thong nhu Tesseract den cac pipeline deep learning nhu EasyOCR, PaddleOCR, VietOCR, va cac mo hinh document AI hien dai nhu PaddleOCR-VL, GLM-OCR, LayoutLMv3. Moi loai OCR co diem manh va gioi han rieng. Voi tai lieu cong van tieng Viet, kho khan khong chi nam o viec doc ky tu, ma con nam o dau tieng Viet, bo cuc hanh chinh, chat luong scan, thu tu doc va trich xuat truong thong tin.

Demo hien tai co gia tri vi no khong phu thuoc vao mot engine duy nhat. He thong cho phep upload tai lieu, tien xu ly anh bang OpenCV, chay nhieu OCR engine, so sanh bang CER/WER, luu ket qua benchmark va hau xu ly bang LayoutLMv3/rule-based extraction. Cach tiep can nay phu hop voi mot de tai nghien cuu ung dung: vua co nen tang ly thuyet OCR, vua co thuc nghiem tren cong cu pho bien, vua co lien he truc tiep voi bai toan so hoa va khai thac cong van tieng Viet.

Neu phai tom tat bang mot cau: OCR hien dai khong con chi la "doc anh thanh chu", ma la mot pipeline gom xu ly anh, phat hien chu, nhan dang chu, hieu bo cuc, danh gia chat luong va trich xuat thong tin co cau truc. Demo da the hien day du tinh than do.
