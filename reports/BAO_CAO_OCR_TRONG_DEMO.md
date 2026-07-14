# Báo cáo chuyên đề: Các OCR được sử dụng trong demo và pipeline xử lý

## 1. Mở đầu

OCR là viết tắt của Optical Character Recognition, nghĩa là nhận dạng ký tự quang học. Nói dễ hiểu, OCR là công nghệ giúp máy tính đọc chữ trong ảnh, bản scan hoặc file PDF dạng hình ảnh, sau đó chuyển thành văn bản có thể sao chép, tìm kiếm, so sánh và xử lý tự động.

Trong demo này, OCR được dùng cho bài toán đọc công văn tiếng Việt. Người dùng upload một file PDF scan hoặc ảnh công văn, hệ thống chuyển tài liệu thành ảnh từng trang, tiền xử lý bằng OpenCV, chạy nhiều OCR engine, sau đó so sánh kết quả bằng các chỉ số như CER, WER, thời gian xử lý và chất lượng văn bản.

Báo cáo này chỉ trình bày các OCR đang được sử dụng trực tiếp trong demo mặc định, gồm:

- Tesseract OCR.
- EasyOCR.
- PaddleOCR kết hợp VietOCR.
- PaddleOCR-VL.

## 2. Mục tiêu OCR trong demo

Mục tiêu chính của demo không chỉ là đọc chữ từ ảnh, mà còn đánh giá xem engine OCR nào phù hợp hơn với tài liệu công văn tiếng Việt. Vì vậy demo được thiết kế theo hướng benchmark nhiều engine thay vì chỉ dùng một OCR duy nhất.

Với mỗi tài liệu đầu vào, hệ thống cần thực hiện các nhiệm vụ sau:

- Chuyển PDF scan thành ảnh từng trang.
- Tạo bản ảnh gốc và bản ảnh đã tiền xử lý OpenCV.
- Chạy các OCR engine đã chọn.
- Thu text, bounding box, confidence và thời gian xử lý.
- So sánh kết quả OCR với file text chuẩn nếu người dùng cung cấp.
- Tính CER và WER để đánh giá lỗi ký tự và lỗi từ.
- Chọn kết quả OCR tốt để đưa sang bước trích xuất trường công văn.

Đối với công văn tiếng Việt, OCR gặp nhiều khó khăn đặc thù:

- Tiếng Việt có dấu, trong đó dấu thanh và dấu nguyên âm rất dễ bị mất khi ảnh mờ hoặc scan chất lượng thấp.
- Văn bản hành chính có nhiều chữ viết tắt như QĐ, TT, NĐ, CP, UBND, NHNN.
- Bố cục công văn có nhiều vùng khác nhau: quốc hiệu, tiêu ngữ, số ký hiệu, ngày ban hành, trích yếu, nội dung, nơi nhận, chữ ký và con dấu.
- Một file PDF có thể có nhiều trang, trong đó chất lượng từng trang không giống nhau.
- OCR có thể đọc đúng từng từ nhưng vẫn sai thứ tự dòng nếu tài liệu có bảng, nhiều cột hoặc header/footer.

## 3. Pipeline OCR trong demo

Pipeline OCR trong demo có thể hiểu là chuỗi xử lý từ file đầu vào đến kết quả benchmark cuối cùng. Các bước chính gồm:

### 3.1. Nhận file đầu vào

Người dùng upload file PDF hoặc ảnh scan công văn. Nếu đầu vào là PDF, hệ thống render từng trang PDF thành ảnh. Nếu đầu vào là ảnh, hệ thống dùng trực tiếp ảnh đó để xử lý.

Đây là bước quan trọng vì OCR không đọc trực tiếp file PDF scan như văn bản thông thường. PDF scan thực chất là ảnh được đóng gói trong file PDF, nên hệ thống phải chuyển từng trang thành ảnh trước khi OCR.

### 3.2. Tiền xử lý ảnh bằng OpenCV

Sau khi có ảnh trang, demo tạo thêm một bản ảnh đã tiền xử lý bằng OpenCV. Mục đích là làm chữ rõ hơn và giảm lỗi OCR.

Các thao tác tiền xử lý chính gồm:

- Chuyển ảnh sang grayscale.
- Phân tích chất lượng trang để biết ảnh đã sạch hay cần xử lý.
- Khử nhiễu.
- Sửa nghiêng trang.
- Tăng tương phản bằng CLAHE.
- Dùng adaptive threshold nếu nền xấu hoặc tương phản thấp.
- Làm nét nhẹ nếu ảnh tương đối sạch.

Điểm đáng chú ý là demo không xử lý mạnh mọi ảnh. Nếu ảnh gốc đã sạch, hệ thống giữ ảnh gần như nguyên bản để tránh làm mất dấu tiếng Việt hoặc làm đứt nét chữ.

### 3.3. Chạy OCR engine

Sau khi có ảnh raw và ảnh OpenCV, demo chạy các engine OCR đã chọn. Mỗi engine trả về text, danh sách box, confidence, thời gian chạy và trạng thái thành công hoặc lỗi.

Các engine chính gồm Tesseract, EasyOCR, PaddleOCR + VietOCR và PaddleOCR-VL. Mỗi engine có cách hoạt động khác nhau nên kết quả cũng khác nhau.

### 3.4. Tính CER và WER

Nếu người dùng cung cấp file text chuẩn, demo sẽ so sánh text OCR với text chuẩn để tính CER và WER.

CER là Character Error Rate, tức tỷ lệ lỗi theo ký tự. Chỉ số này phù hợp để đánh giá lỗi dấu tiếng Việt, lỗi sai chữ cái hoặc lỗi thiếu ký tự.

WER là Word Error Rate, tức tỷ lệ lỗi theo từ. Chỉ số này cho biết kết quả OCR có giữ đúng các từ và cụm từ trong văn bản hay không.

CER/WER càng thấp thì kết quả OCR càng tốt. Nếu không có file text chuẩn, demo vẫn chạy OCR nhưng không thể đánh giá lỗi định lượng bằng CER/WER.

### 3.5. Hậu xử lý và trích xuất trường

Sau khi có kết quả OCR, demo chọn kết quả tốt để đưa sang bước trích xuất trường công văn. Các trường thường cần lấy gồm số ký hiệu, ngày ban hành, trích yếu, cơ quan ban hành, nơi gửi, nơi nhận và loại văn bản.

Nếu có model LayoutLMv3 đã fine-tune, hệ thống có thể dùng model này để trích xuất trường theo text và vị trí box. Nếu chưa có model hoặc model không sẵn sàng, demo dùng rule/regex fallback để vẫn có kết quả.

## 4. Tesseract OCR

### 4.1. Khái quát

Tesseract là một engine OCR nguồn mở lâu đời và rất phổ biến. Nó thường được dùng làm baseline trong các hệ thống OCR vì dễ triển khai, chạy offline và hỗ trợ nhiều ngôn ngữ, trong đó có tiếng Việt.

Trong demo, Tesseract được dùng để đọc ảnh công văn sau khi hệ thống đã chuẩn bị ảnh đầu vào. Engine này phù hợp với tài liệu scan rõ, nền trắng, chữ in đen, bố cục không quá phức tạp.

### 4.2. Nguyên lý hoạt động

Tesseract thực hiện các bước chính:

- Phân tích ảnh đầu vào.
- Xác định vùng chữ, dòng chữ và từ.
- Nhận dạng nội dung chữ bằng mô hình OCR.
- Trả về văn bản, bounding box và confidence.

Các phiên bản Tesseract mới sử dụng mô hình LSTM để nhận dạng chuỗi ký tự tốt hơn so với cách OCR truyền thống chỉ dựa vào tách ký tự và so khớp mẫu.

### 4.3. Cách demo sử dụng Tesseract

Trong demo, Tesseract được gọi thông qua thư viện pytesseract. Code tự tìm Tesseract binary trong máy, trong PATH hoặc trong các thư mục cài đặt phổ biến. Demo cũng tìm thư mục tessdata để dùng dữ liệu ngôn ngữ tiếng Việt và tiếng Anh.

Khi chạy, Tesseract trả về dữ liệu dạng word-level, gồm:

- Text từng từ.
- Tọa độ bounding box.
- Confidence của từng từ.
- Thời gian xử lý.

Nếu máy thiếu dữ liệu tiếng Việt, demo có cơ chế fallback sang tiếng Anh để quá trình demo không dừng hoàn toàn.

### 4.4. Ưu điểm

- Chạy local/offline, không cần Internet.
- Nhẹ hơn nhiều OCR deep learning.
- Dễ tích hợp vào Python qua pytesseract.
- Phù hợp làm engine baseline để so sánh.
- Có hỗ trợ tiếng Việt thông qua language data.

### 4.5. Hạn chế

- Cần cài Tesseract binary ngoài Python.
- Dễ sai khi ảnh mờ, nghiêng hoặc nền xấu.
- Có thể sai dấu tiếng Việt.
- Chưa mạnh với tài liệu có bố cục phức tạp, bảng biểu hoặc nhiều cột.
- Thứ tự đọc có thể sai nếu layout của trang không đơn giản.

## 5. EasyOCR

### 5.1. Khái quát

EasyOCR là thư viện OCR dựa trên deep learning. Nó hỗ trợ nhiều ngôn ngữ và có thể xử lý cả ảnh scan lẫn ảnh chụp trong điều kiện thực tế.

Trong demo, EasyOCR được dùng như một engine OCR học sâu để so sánh với Tesseract và PaddleOCR. EasyOCR thường có lợi thế hơn Tesseract khi ảnh đầu vào không quá sạch hoặc có đặc điểm giống ảnh chụp.

### 5.2. Nguyên lý hoạt động

EasyOCR thường gồm hai phần chính:

- Text detection: phát hiện vùng có chữ trong ảnh.
- Text recognition: nhận dạng nội dung trong từng vùng chữ.

Nhờ dùng mô hình học sâu, EasyOCR có khả năng phát hiện chữ linh hoạt hơn trong nhiều bối cảnh. Engine này không chỉ phụ thuộc vào quy tắc hình học đơn giản mà học từ dữ liệu ảnh chữ.

### 5.3. Cách demo sử dụng EasyOCR

Trong demo, EasyOCR được chạy trong một worker riêng. Điều này giúp giảm rủi ro xung đột thư viện và giúp hệ thống kiểm soát timeout tốt hơn.

Kết quả EasyOCR trả về gồm:

- Text OCR.
- Danh sách box.
- Confidence.
- Trạng thái chạy.
- Lỗi nếu thiếu package hoặc worker không chạy được.

Nếu máy chưa cài EasyOCR, demo sẽ báo trạng thái lỗi hoặc skipped thay vì làm sập toàn bộ ứng dụng.

### 5.4. Ưu điểm

- Dễ dùng trong Python.
- Hỗ trợ nhiều ngôn ngữ.
- Có detector và recognizer học sâu.
- Có thể hoạt động tốt với ảnh chụp hoặc ảnh scan không quá lý tưởng.
- Trả về box và confidence thuận tiện cho benchmark.

### 5.5. Hạn chế

- Cài đặt nặng hơn Tesseract vì phụ thuộc PyTorch.
- Chạy trên CPU có thể chậm.
- Lần đầu chạy có thể phải tải model.
- Vẫn có thể sai dấu tiếng Việt.
- Với tài liệu bố cục phức tạp, thứ tự text đầu ra có thể chưa đúng hoàn toàn.

## 6. PaddleOCR kết hợp VietOCR

### 6.1. Khái quát

PaddleOCR là bộ OCR mạnh thuộc hệ sinh thái PaddlePaddle. Nó có khả năng phát hiện chữ, nhận dạng chữ và hỗ trợ nhiều ngôn ngữ, trong đó có tiếng Việt.

VietOCR là mô hình OCR tập trung vào nhận dạng tiếng Việt. Trong demo, VietOCR không đứng riêng như một engine độc lập mà được dùng để refine kết quả của PaddleOCR khi cấu hình cho phép.

Sự kết hợp này có ý nghĩa thực tế: PaddleOCR mạnh ở phát hiện vùng chữ, còn VietOCR có thể cải thiện phần nhận dạng tiếng Việt, nhất là khi được fine-tune trên dữ liệu công văn.

### 6.2. Nguyên lý hoạt động

Pipeline PaddleOCR + VietOCR trong demo có thể hiểu như sau:

- PaddleOCR nhận ảnh đầu vào.
- PaddleOCR phát hiện các vùng chữ trong ảnh.
- PaddleOCR nhận dạng sơ bộ nội dung từng vùng.
- Nếu bật refine, demo crop từng vùng chữ theo bounding box.
- VietOCR nhận dạng lại từng crop.
- Hệ thống ghép kết quả thành text cuối cùng.

Cách này giúp tận dụng điểm mạnh của cả hai công cụ. Tuy nhiên, nếu box do PaddleOCR phát hiện bị thiếu chữ hoặc thiếu dấu, VietOCR cũng khó sửa được vì crop đầu vào đã thiếu thông tin.

### 6.3. Cách demo sử dụng PaddleOCR + VietOCR

Trong demo, engine này có tên `paddle_vietocr`. PaddleOCR được khởi tạo với ngôn ngữ tiếng Việt. Một số thành phần như orientation classify hoặc document unwarping được tắt để giảm độ phức tạp khi chạy local.

Engine cũng chạy trong subprocess worker riêng để tránh xung đột giữa PaddlePaddle, PyTorch và các thư viện liên quan đến CPU/GPU.

Nếu bật VietOCR refine, hệ thống sẽ gọi worker riêng để nhận dạng lại các crop. Repo có model VietOCR fine-tuned cho công văn, cho thấy demo hướng đến tối ưu hóa cho tài liệu hành chính tiếng Việt.

### 6.4. Ưu điểm

- PaddleOCR có detector mạnh và phù hợp nhiều loại tài liệu.
- Hỗ trợ tiếng Việt.
- VietOCR có lợi thế với văn bản tiếng Việt có dấu.
- Có thể fine-tune VietOCR theo dữ liệu công văn.
- Kết quả có box và confidence để phục vụ benchmark.

### 6.5. Hạn chế

- Cài đặt nặng hơn Tesseract và EasyOCR.
- Phụ thuộc PaddlePaddle, dễ gặp vấn đề version trên Windows.
- Chạy CPU có thể chậm, đặc biệt khi bật VietOCR refine.
- Kết quả refine phụ thuộc chất lượng bounding box của PaddleOCR.
- Nếu tài liệu nhiều trang, thời gian xử lý có thể tăng rõ rệt.

## 7. PaddleOCR-VL

### 7.1. Khái quát

PaddleOCR-VL là hướng OCR/document parsing hiện đại hơn so với OCR chỉ đọc từng dòng chữ. Nó có mục tiêu không chỉ nhận dạng text mà còn hiểu bố cục tài liệu và có thể sinh kết quả dạng markdown, json hoặc cấu trúc layout.

Trong demo, PaddleOCR-VL đại diện cho nhóm OCR hiện đại có khả năng xử lý tài liệu phức tạp hơn, đặc biệt khi văn bản có bảng, nhiều khối nội dung hoặc cần giữ cấu trúc.

### 7.2. Nguyên lý hoạt động

PaddleOCR-VL kết hợp khả năng nhìn ảnh và xử lý ngôn ngữ. Thay vì chỉ trả ra danh sách từ, nó có thể phân tích tài liệu ở mức cao hơn:

- Nhận diện text.
- Nhận diện vùng layout.
- Trả kết quả theo dạng markdown hoặc json.
- Hỗ trợ hiểu cấu trúc tài liệu tốt hơn OCR truyền thống.

Điều này phù hợp với hướng document AI, nơi mục tiêu không chỉ là đọc chữ mà còn hiểu tài liệu.

### 7.3. Cách demo sử dụng PaddleOCR-VL

Trong demo, PaddleOCR-VL có thể chạy theo hai cách:

- Dùng command cấu hình qua `PADDLEOCR_VL_CMD`.
- Dùng Python API `PaddleOCRVL` nếu môi trường hỗ trợ.

Engine sẽ đọc các file output như markdown, text hoặc json để trích nội dung. Kết quả sau đó được chuẩn hóa về text và boxes để đưa vào dashboard benchmark.

### 7.4. Ưu điểm

- Phù hợp với tài liệu có bố cục phức tạp.
- Có thể trả kết quả gần với cấu trúc tài liệu.
- Hữu ích khi cần giữ bảng, block hoặc markdown.
- Đại diện cho xu hướng OCR hiện đại kết hợp document understanding.

### 7.5. Hạn chế

- Cài đặt và chạy nặng.
- Có thể mất nhiều thời gian trên CPU.
- Cần version PaddleOCR/PaddlePaddle phù hợp.
- Không nên bật cho mọi lần demo nếu chỉ cần kết quả nhanh.
- Với file nhiều trang, thời gian xử lý có thể rất dài.

## 8. So sánh các OCR trong demo

| Engine | Vai trò trong demo | Điểm mạnh | Hạn chế chính |
|---|---|---|---|
| Tesseract | Baseline OCR local/offline | Nhẹ, dễ chạy, phù hợp scan sạch | Dễ sai với ảnh xấu và layout phức tạp |
| EasyOCR | OCR deep learning tổng quát | Hỗ trợ nhiều ngôn ngữ, tốt với ảnh không quá sạch | Nặng hơn, CPU có thể chậm |
| PaddleOCR + VietOCR | OCR tiếng Việt nâng cao | PaddleOCR detect tốt, VietOCR hỗ trợ tiếng Việt | Cài đặt nặng, refine chậm |
| PaddleOCR-VL | OCR/document parsing hiện đại | Hiểu layout tốt hơn, có markdown/json | Rất nặng, cần cấu hình môi trường |

Nếu cần demo nhanh, có thể ưu tiên Tesseract, EasyOCR hoặc PaddleOCR. Nếu cần phân tích sâu về bố cục tài liệu, có thể bật PaddleOCR-VL nhưng cần chuẩn bị thời gian và tài nguyên máy.

## 9. CER/WER trong demo

CER và WER là hai chỉ số quan trọng để đánh giá kết quả OCR.

CER đo tỷ lệ lỗi ký tự. Ví dụ nếu OCR làm mất dấu tiếng Việt hoặc nhận nhầm chữ cái, CER sẽ tăng. Chỉ số này rất phù hợp với tiếng Việt vì lỗi dấu thường chỉ là một ký tự nhỏ nhưng ảnh hưởng lớn đến chất lượng văn bản.

WER đo tỷ lệ lỗi từ. Nếu OCR nhận sai, thiếu hoặc thừa từ, WER sẽ tăng. Chỉ số này cho biết văn bản OCR có dùng được ở mức nội dung hay không.

Trong demo, người dùng có thể upload file text chuẩn ở định dạng txt, md, csv, json, doc hoặc docx. Tuy nhiên, dùng file txt là ổn định nhất vì hệ thống đọc trực tiếp văn bản thuần. File docx cũng đọc được, còn file doc cần Microsoft Word trên máy để trích xuất nội dung.

## 10. Nhận xét chung

Các OCR trong demo có vai trò bổ sung cho nhau. Tesseract phù hợp làm mốc so sánh vì nhẹ và dễ chạy. EasyOCR thể hiện hướng OCR học sâu dễ sử dụng. PaddleOCR + VietOCR phù hợp hơn với mục tiêu tiếng Việt và công văn. PaddleOCR-VL mở rộng demo sang hướng OCR có hiểu bố cục tài liệu.

Việc chạy nhiều engine giúp demo không phụ thuộc vào một công cụ duy nhất. Kết quả benchmark cho phép đánh giá engine nào nhanh hơn, engine nào chính xác hơn và tiền xử lý OpenCV có cải thiện chất lượng OCR hay không.

Đối với bài toán công văn tiếng Việt, lựa chọn OCR tốt nhất không chỉ dựa vào độ chính xác text toàn văn, mà còn phụ thuộc vào khả năng giữ dấu tiếng Việt, đọc đúng số ký hiệu, ngày ban hành, trích yếu và các vùng thông tin quan trọng.

## 11. Kết luận

Demo hiện tại sử dụng bốn nhóm OCR chính: Tesseract, EasyOCR, PaddleOCR kết hợp VietOCR và PaddleOCR-VL. Mỗi engine có ưu điểm và hạn chế riêng. Tesseract nhẹ và dễ chạy, EasyOCR linh hoạt với nhiều loại ảnh, PaddleOCR + VietOCR phù hợp tiếng Việt hơn, còn PaddleOCR-VL đại diện cho hướng OCR hiện đại có khả năng hiểu bố cục tài liệu.

Pipeline của demo thể hiện đúng cách xây dựng một hệ thống OCR thực tế: nhận file, render PDF thành ảnh, tiền xử lý ảnh, chạy nhiều OCR engine, tính CER/WER, so sánh kết quả và đưa output sang bước trích xuất trường công văn. Nhờ vậy, demo không chỉ minh họa lý thuyết OCR mà còn có giá trị thực nghiệm cho bài toán số hóa và khai thác công văn tiếng Việt.
