# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:6/11/2026**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab; `rank`=1 ; `score`=0.510915; `taxonomy_name`=ImageNet-1K
2. Record này nói về toàn bộ vật thể trong ảnh. Vì nó không khoanh vùng 1 vật thể cụ thể nào, mà đánh giá toàn bộ khung hình có xác suất cao nhất thuộc về lớp "cab". Thông số score thể hiện mức độ tin cậy của mô hình trên tổng thể bức ảnh.
3. Danh sách tên lớp (class list) do đâu mà có: Do đi kèm taxonomy/checkpoint. Vì khi mô hình được huấn luyện trên ImageNet-1K, danh sách nhãn chuẩn của taxonomy
ImageNet-1K đó được gán cứng và đóng gói trực tiếp vào file trọng số.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? Vì nếu chỉ lưu class_name="cab", dữ liệu có thể gây nhầm lẫn khi nhiều taxonomy sử dụng cùng tên lớp nhưng có cách đánh số hoặc định nghĩa lớp khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp? Guideline cần quy định rõ tiêu chí chọn một nhãn duy nhất để đại diện cho toàn bộ ảnh, tránh việc mỗi người gán nhãn chọn 1 vật thể hoặc chủ đề khác nhau.
6. Vì sao không được lấy `score` của model làm ground truth? Vì score là điểm xác suất dự đoán do mô hình tính toán, không phải nhãn chuẩn được con nguòi xác nhận.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel? Góc Nằm tại tọa độ pixel (385,33,69,24), nghĩa là cách mép trái của bức ảnh 385.33 px và cách mép trên của bức ảnh 69.24 px.
3. Chọn hai ngưỡng ( 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction? 
Ở ngưỡng 20, số lượng prediction sẽ là 17, ở ngưỡng 60 số lượng prediction sẽ là 6.
- Threshold 0.20: Có 17 prediction, giữ lại nhiều vật thể hơn, bao gồm spoon, potted plant, dining table và bottle.
- Threshold 0.60: Chỉ có 6 prediction, gồm person, bowl và oven.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Reviewer sẽ phải:
- Kiểm tra nhiều prediction hơn.
- Xác định các box phát hiện đúng và phát hiện sai.
- Kiểm tra nhãn class có chính xác không.
- Phát hiện box bị thiếu, box dư hoặc box chồng lặp.
- Xử lý các trường hợp khó như vật thể nhỏ, bị che khuất hoặc confidence thấp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Bouding Box Annotation Rule 
Người dán nhãn sẽ phải vẽ box hình chữ nhật bao sát phần nhìn thấy của vật thể cần gán nhãn. Box phải chứa toàn bộ phần vật thể nhìn thấy, không bao gồm quá nhiều nền xung quanh. Nếu vật thể bị che khuất, chỉ vẽ quanh phần nhìn thấy theo guideline. Nếu vật thể bị che khuất, chỉ vẽ quanh phần nhìn thấy theo guideline. Nếu vật thể bị cắt bởi mép ảnh, vẽ box đến sát mép ảnh và không tự suy đoán phần nằm ngoài ảnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Trên ảnh `kitchen` **một** vật bị che khuất là bàn gỗ ở góc dưới bên trái ảnh, bị cắt bởi mép trái và mép dưới của hình ảnh. Trường hợp này tôi sẽ đưa vào guideline vì đây là tình huống có thể gặp ở nhiều ảnh khác nhau, cần quy định thống nhất để người dán nhãn thực hiện.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001; `class_name`= person; `score`= 0.899318; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon mô tả đường viền và hình dạng thực tế của vật thể chi tiết hơn box. Box chỉ bao quanh vật thể bằng hình chữ nhật, còn polygon có thể bám sát các phần lồi lõm, giúp xác định vùng pixel thuộc về vật thể chính xác hơn.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
`instance_id` là mã định danh riêng cho từng đối tượng cụ thể trong ảnh.  Nó **không** phải class ID vì class ID cho biết lớp chung (tức cho biết vật thể thuộc loại gì), còn `instance_id` cho biết đó là vật thể cụ thể nào. Còn `instance_id` không phải tracking ID vì `instance_id` chỉ nhận diện vật thể trong ảnh hiện tại, không dùng để theo dõi vật thể đó di chuyển qua nhiều khung hình video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Vẽ polygon sát theo đường viền phần vật thể nhìn thấy trong ảnh, không bao gồm nền hoặc vật thể khác; khi bị che khuất, chỉ vẽ phần nhìn thấy được, không tự suy đoán phần bị che.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Trên ảnh `kitchen`,**một** chỗ biên mờ / hai vật chạm nhau / bị che là người đứng trước khu vực bếp. Vì phần người đứng gần các dụng cụ bếp và khu vực phía sau có chỗ bị che khuất, biên giữa người và nền không hoàn toàn rõ. Với tình huống này sẽ dùng guideline để vẽ mask theo phần người nhìn thấy được. Vì trường hợp vật thể chạm nhau hoặc che khuất một phần là tình huống phổ biến đã được định nghĩa rõ trong tài liệu hướng dẫn gán nhãn (guideline). Gán nhãn viên chỉ cần áp dụng quy tắc chuẩn đã thống nhất: vẽ polygon ôm sát phần viền thực tế nhìn thấy, không tự ý suy đoán hay cố gắng vẽ bù phần bị khuất của vật thể bị che. Do ranh giới vật thể vẫn có thể phân biệt được dựa trên độ tương phản hoặc quy tắc hiển thị thông thường, người gán nhãn hoàn toàn xử lý theo guideline mà không cần escalation lên cấp trên
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh (Image Classification) | Nhãn dạng tên lớp (Class label, ví dụ: kitchen, street). | Dẫn chứng JSON/PNG(kitchen): Ảnh chứa nhiều đối tượng đan xen (bếp, nồi, bàn, người), dẫn đến file JSON ghi nhận nhãn bao trùm có thể chưa phản ánh hết chi tiết nhỏ bên trong. | Gán nhãn tên lớp chính xác cho toàn bộ bức ảnh. | Kiểm tra xem nhãn phân loại có phản ánh đúng đối tượng chính hay không. |
| Phát hiện vật thể (Object Detection) | Tọa độ hộp bao [xmin, ymin, xmax, ymax] kèm tên lớp. | Dẫn chứng JSON/PNG: Trong file JSON/ảnh, các hộp bao (bounding box) của `bowl` hoặc `oven` bị chồng chéo lên nhau hoặc lẹm vào ranh giới bàn `dining table`. | Vẽ khung chữ nhật (bounding box) bao phủ toàn bộ vùng hiển thị của vật thể. | Kiểm tra độ khít của bounding box và xem có bỏ sót vật thể nào không. |
| Instance segmentation | Tọa độ chuỗi đa giác (Polygon mask) bám sát viền đối tượng. | Dẫn chứng JSON/PNG: Trên ảnh, biên vật thể bị mờ hòa vào nền tối hoặc các vật thể chạm sát nhau (như đồ vật trên bàn), khiến tọa độ polygon trong JSON dễ bị lẹm hoặc tràn viền. | Vẽ polygon ôm sát phần viền thực tế nhìn thấy theo đúng quy tắc biên mask. | Kiểm tra viền polygon có bám sát đối tượng, không lẹm nền và không tự suy đoán phần che khuất. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Quy tắc bảo vệ dữ liệu của bạn cho bài lab này Chỉ sử dụng dữ liệu mẫu được cung cấp trong phạm vi bài lab cho mục đích học tập, tuyệt đối không sao chép, lưu trữ ngoài luồng hay làm rò rỉ dữ liệu ra bên ngoài.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Dừng ngay thao tác và báo cáo trực tiếp cho Giảng viên hướng dẫn hoặc Quản lý dự án.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
