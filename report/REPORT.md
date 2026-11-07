# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:** Python: 3.13.16, PyTorch: 2.11.0+cu130, Ultralytics: 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

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

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này đại diện cho kết quả phân loại của toàn bộ bức ảnh, không phải một vật thể riêng biệt, vì record tham chiếu cùng 1 hình ảnh, task thể hiện tác vụ phân loại ảnh, các thông số rank, class_id, class_name và score thể hiện dự đoán có thứ hạng khác nhau của mô hình và không có field xác định vị trí record
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm taxonomy/checkpoint
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Cần cả class_id, class_name và taxonomy_name vì 3 field này đại diện cho 3 điều khác nhau:  ID là mã định danh máy đọc được, hiệu quả cho máy, Class name là nhãn người đọc được, giúp người dán nhãn đọc và ghi nhớ dễ hơn, Taxonomy định dạng hệ thống phân loại hay phân cấp loại, giúp giữ được sự đồng nhất
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guildeline cần nói rõ rằng: Cần chú thích mọi object xác định được nằm trong class list được chấp thuận
+ Mỗi object riêng biệt nhận 1 chú thích của chính nó
+ Assign 1 class phù hợp cho mỗi object
+ Dùng class cụ thể nhất hiên có cho 1 object
+ Không assign nhiều class không liên quan cho 1 object
+ Không tạo ra class cho object năm ngoài taxonomy
+ Đánh dấu object không rõ hay bị che khuất cho đánh giá dựa theo project policy
6. Vì sao không được lấy `score` của model làm ground truth?
- Score của model không hề có nghĩa là object đó đúng chính xác là label đó, nó chỉ có nghĩa là model assign điểm confidence cho dự đoán của nó. Model có thể sai một cách tự tin, Confidence dựa trên model, checkpoint khác nhau có thể tạo ra dự đoán khác nhau cho cùng 1 bức ảnh

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= oven ; `score`= 0.686792 ; `bbox_xyxy`=[0.11, 187.54, 195.96, 292.81] ; `bbox_width`= 195.85 ; `bbox_height`= 105.26
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên trái tại (0.11, 187.54), góc dưới phải tại (195.96, 292.81), rộng 195.85 pixel, cao 105.26 pixel
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Ngưỡng 0.2: 17 prediction
Ngưỡng 0.5=6: 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Reviewer cần phải review thêm prediction mới, đặc biệt là những prediction với điểm confidence thấp, bới ngưỡng nhỏ hơn cho phép YOLO giữ lại thêm các prediction, kể cả những cái có thể sai
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Vẽ 1 bounding box chữ nhật chặt xung quanh mỗi object xác định được, bao gồm toàn bộ object và giảm thiểu background không liên quan, không kết hợp nhiều object vào 1 box và đảm bảo mỗi box được assign class đúng. Không đoán biên ẩn, escalate những case không chắc chắn để review
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vật bị cắt mép: Person 0.61
Tôi sẽ chọn viết Guideline bởi tay người là thứ hoàn toàn rõ ràng, có thể viết guideline chung để annotate phần cánh tay đó chính là "person"
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=(446, 70), (445, 71), ()
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon cung cấp thêm silhouette vât thể (các hình khổi khác thường thay vì giả định tất cả là hình chữ nhật), chính xác biên, thông tin hình khối và tách biệt đối tượng
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id không phải là class_id, vì nó là mã định danh cho 1 đối tượng cụ thể. Nó cũng không phải là tracking ID vì tracking ID kết nối cùng 1 vật thể xuyên suốt các video frame, trong khi instance_id chỉ định danh 1 đối tượng innotate duy nhất phía trong dataset scope liên quan
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Trace những khu vực bên ngoài của từng object gắn chặt nhất có thể. Bao gồm tất cả những phần nhìn thấy được, loại bỏ background và các object gần kề, giữ object riêng biệt trong mask riêng biệt cả khi chúng chạm hay overlap. Không đoán biên ẩn, escalate những case không chắc chắn để review
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
2 vật chạm nhau: bowl 0.53 và bowl 0.74
Tôi sẽ chọn viết Guideline vì 2 chiếc bát này hoàn toàn có thể nhìn thấy được, không bị mờ và có thể hướng dẫn annotate label 'bowl'
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một hoặc nhiều class label cho toàn bộ ảnh. Đơn vị nhãn: ảnh |  - Ảnh classification_top5.png chứa nhiều phương tiện khác nhau nhưng mô hình chỉ đưa ra các dự đoán ở cấp độ toàn ảnh. <br> - JSON có task: image_classification, cả 5 records cùng sample_id: traffic nhưng có rank, class_name và score khác nhau. Điều này chứng minh các records là 5 dự đoán cho cùng một bức ảnh, không phải 5 object riêng biệt. <br> - Không có ground truth  | Chọn class theo taxonomy và quy tắc phân loại ảnh. Không tự ý thêm class ngoài list | Kiểm tra class đúng nội dung ảnh không, quy tắc chọn class được áp dụng nhất quán không |
| Phát hiện vật thể | Mỗi object có 1 class và 1 bounding box. Đơn vị nhãn: từng object |  - Ảnh detection_predictions.png thể hiện các bounding box bao quát cả khu vực background và object xung quanh, như 2 object bowl bị overlap box với nhau hay object person có quá nhiều background xung quanh. <br> - 1 vật thể person bị cắt mép, chỉ hiển thị tay trong detection_predictions.png. <br> - 2 class bowl có bbox chồng lên nhau trong detection_predictions.png. <br> - Không có ground truth trong detection_predictions.json sample_id kitchen| Vẽ bbox sát object, gán đúng class, tạo annotation cho từng object. Escalate nếu không thể xác định class | Kiểm tra số lượng object, class, vị trí và kích thước bbox, phát hiện các object bị bỏ sót, gán trùng hoặc gán sai |
| Instance segmentation | Mỗi object có 1 class và 1 mask/polygon riêng. Đơn vị nhãn: từng instance với mask/polygon | - Dining Table bị overlap với vật thể không liên quan trong segmentation_predictions.png. <br> - Mask spoon phía bên trái segmentation_predictions.png cả 2 object thay vì chỉ mask từng object một. <br> - Object bó rau treo phía trên class sai là potted plant phía bên phải trên cùng segmentation_predictions.png | Vẽ mask theo biên nhìn thấy của từng object. Không đoán phần bị che nếu không xác định được chắc chắn. Escalate nếu không xác định được biên | Kiểm tra mask có theo đúng silhouette không, dán class đúng không, lẫn background hay chồng lên các vật thể không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Chỉ sử dụng đoạn code learning và dữ liệu được cung cấp, không thêm, xóa hay sửa nếu không có yêu cầu gì thêm, không tự ý chia sẽ, sao chép hoặc tiết lộ dữ liệu cho người không có thẩm quyền
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Khi gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và cáo cáo cho Lab Coach để escalate những phần dữ liệu bị dán nhãn sai

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ x ] `REPORT.md` (file này đã điền)
- [ x ] `day1_lab_outputs/classification_predictions.json`
- [ x ] `day1_lab_outputs/detection_predictions.json`
- [ x ] `day1_lab_outputs/segmentation_predictions.json`
- [ x ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ x ] `day1_lab_outputs/visuals/classification_top5.png`
- [ x ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ x ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ x ] Ô validation cuối notebook báo `PASS`
- [ x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ x ] Đã nộp **link repository** trên LMS