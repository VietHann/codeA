# Ý tưởng hiện tại: SepFormer + Transformer JEPA

## 1. Mục tiêu và phạm vi

Giữ nguyên SepFormer làm mô hình tách nguồn chính, đồng thời thêm một mục tiêu huấn luyện phụ: **dự đoán biểu diễn của nguồn sạch tại các vị trí bị che trên latent nguồn ước lượng**.

Cả separator và hai sequence model của nhánh JEPA hiện dùng Transformer. JEPA chỉ tham gia huấn luyện; inference vẫn là encoder → SepFormer MaskNet → decoder, không cần nguồn sạch, context/predictor hay EMA target.

Đây là một thiết kế **JEPA-inspired có giám sát bằng clean sources**, không phải tuyên bố tái hiện nguyên bản một phương pháp JEPA cụ thể hay một quy trình self-supervised không cần nhãn nguồn.

Cấu hình được mô tả: [hparams/LibriMix/sepformer.yaml](hparams/LibriMix/sepformer.yaml).

## 2. Separator chính không thay đổi

Với mixture `x` và các nguồn tham chiếu `s₁, …, sS`:

- Waveform encoder tạo đặc trưng `W = Encoder(x)`.
- SepFormer MaskNet tạo mask cho từng nguồn `Mᵢ = MaskNet(W)ᵢ`.
- Latent nguồn ước lượng là `Hᵢ = W ⊙ Mᵢ`.
- Decoder tạo waveform `ŝᵢ = Decoder(Hᵢ)`.

Các block `SBtfintra`, `SBtfinter` vẫn là Transformer của SepFormer; `MaskNet` vẫn là `Dual_Path_Model` với 2 dual-path layers và chunk size 250. Không thay separator thành DPRNN hoặc DPMamba.

**Điểm nối JEPA là `Hᵢ`: latent sau khi nhân mask của separator, trước decoder.** Đây không phải hidden state của một attention layer bên trong SepFormer, cũng không phải waveform đã decode.

Ký hiệu shape:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| Mixture | `[B,T]` | Waveform đầu vào |
| Clean sources | `[B,T,S]` | Waveform nguồn tham chiếu |
| Mixture latent | `[B,C,F]` | Đặc trưng encoder |
| Source latents đưa vào JEPA | `[B,S,C,F]` | Latent nguồn sau mask, được PIT-align |

Trong cấu hình này: `C = N_encoder_out = 256`, mặc định `S = 2`. `F` là số frame sau convolution, không phải số sample waveform.

## 3. PIT alignment trước khi tính JEPA

Thứ tự nguồn đầu ra của separator không cố định. Nếu so sánh latent nguồn ước lượng với clean source chỉ theo index, JEPA có thể học nhầm cặp speaker.

Code giải quyết bằng cách:

1. Tìm permutation tối ưu giữa waveform dự đoán và nguồn tham chiếu theo SI-SNR.
2. Tìm permutation trong `no_grad`; không đạo hàm qua bước chọn thứ tự.
3. Áp dụng cùng permutation đó lên latent `H`.
4. Đưa latent đã sắp thứ tự và clean sources vào JEPA.

Việc chọn permutation không có gradient, nhưng phép sắp lại latent **không detach latent**: JEPA vẫn truyền gradient về encoder và MaskNet.

## 4. Online branch: masked source prediction

Mỗi nguồn được xử lý như một sequence riêng bằng cách gộp batch và source: `[B,S,C,F]` → `[B×S,C,F]`.

### 4.1. Chọn các frame hợp lệ và sinh mask

- Frame chỉ hợp lệ khi toàn bộ receptive field convolution của nó nằm trong vùng waveform hợp lệ.
- Padding và khoảng trống do augmentation không được dùng làm target.
- Sinh mask theo các span liên tiếp dài 16 frame, với xác suất chọn span 0.5.
- Code bảo đảm có vị trí được giám sát nếu có frame hợp lệ, đồng thời giữ ít nhất một frame nhìn thấy nếu có từ hai frame hợp lệ trở lên.

`mask_ratio = 0.5` là xác suất chọn span; tỷ lệ frame thực tế của từng sample không bắt buộc đúng 50%.

### 4.2. Context Transformer

Các latent bị che và latent không hợp lệ được đặt bằng 0 trước context model:

`H_visible = H_aligned với các vị trí bị mask/không hợp lệ được đặt bằng 0`

`Z = ContextTransformer(H_visible)`

Context nhận và trả sequence `[B×S,F,256]`. Nó là một Transformer riêng, không chia sẻ trọng số với `SBtfintra` hoặc `SBtfinter`.

### 4.3. Projection, mask token và Predictor Transformer

Sau context:

1. `input_proj`: Conv1d kernel 1, chuyển 256 → 128 chiều.
2. Thay biểu diễn tại các vị trí bị mask bằng một **learnable mask token** 128 chiều.
3. Đặt vùng không hợp lệ về 0.
4. Predictor Transformer xử lý sequence 128 chiều.
5. `output_proj`: Conv1d kernel 1, chuyển 128 → 256 chiều để so sánh với target.

Điểm quan trọng: context nhìn thấy các vị trí bị che dưới dạng **zero latent**; predictor nhìn thấy chúng dưới dạng **mask token**. Đây là hai thao tác khác nhau và đều được giữ nguyên.

Context/predictor không dùng causal mask, nên có thể khai thác context ở cả trước và sau vị trí cần dự đoán.

## 5. Target branch: biểu diễn nguồn sạch bằng EMA

Target branch gồm:

- `target_encoder`: bản sao đóng băng của waveform encoder.
- `target_context`: bản sao đóng băng của online context Transformer.

Nó nhận waveform clean source, không nhận latent nguồn ước lượng:

`U = TargetEncoder(clean_source)`

`Y_raw = TargetContextTransformer(U)`

`Y = LayerNorm(Y_raw, theo chiều channel tại mỗi frame)`

Target branch không áp dụng JEPA mask lên clean latents và chạy trong `no_grad`, luôn ở eval mode. Không có target predictor.

Ở đây, “clean source” là nguồn tham chiếu của **batch hiện tại sau các augmentation/cắt tín hiệu liên quan**, không nhất thiết là waveform gốc chưa augmentation từ file.

### Cập nhật EMA

Sau khởi tạo lại model của recipe, target được đồng bộ từ online encoder/context trước khi recover checkpoint. Sau mỗi optimizer step thành công:

`θ_target ← m × θ_target + (1 − m) × θ_online`

Momentum bắt đầu ở 0.996 và được cosine-anneal tới 1.0 theo số lần cập nhật EMA. Với cấu hình hiện tại, `anneal_steps = 200 × (20000 // 1) = 4,000,000`.

- EMA áp dụng cho waveform encoder và context, không áp dụng cho predictor.
- Target không học bằng backpropagation.
- Batch bị bỏ qua không cập nhật EMA; trong AMP, code kiểm tra việc scaler có bỏ optimizer step hay không.

## 6. Loss và luồng gradient

### Loss separation

Loss chính là negative SI-SNR với PIT trên waveform.

### Loss JEPA

Tại mỗi frame, tính mean squared error giữa prediction 256 chiều và target 256 chiều:

`frame_loss = mean_channel((prediction − target)²)`

Chỉ tính loss trên frame thỏa cả ba điều kiện:

- Được JEPA mask chọn.
- Hợp lệ theo waveform/receptive field.
- Clean latent đủ hoạt động: norm theo channel của `target_encoder` output lớn hơn `1e-4`.

Mỗi sample được chuẩn hóa bằng tổng số frame đủ điều kiện trên tất cả nguồn. Nếu không có frame đủ điều kiện, contribution là 0.

### Loss tổng

`L_total = L_separation + 0.1 × L_JEPA`

Công thức trên đi kèm logic lọc sample hiện có: khi `threshold_byloss` bật, loss separation và JEPA được lấy trên các sample có waveform loss lớn hơn threshold (`-30` trong config). Nếu không còn sample đủ điều kiện, batch không tạo một bước học chỉ bằng JEPA.

### Thành phần nhận gradient từ JEPA

- Online waveform encoder: có.
- SepFormer MaskNet: có, thông qua latent nguồn sau mask.
- Context Transformer, predictor Transformer, input/output projection và mask token: có đường gradient, tùy vị trí mask và dữ liệu hợp lệ.
- Decoder: không nhận gradient trực tiếp từ JEPA; vẫn được học bằng loss separation.
- Target encoder/context: không; chỉ cập nhật bằng EMA.

## 7. Cấu hình Transformer của nhánh JEPA hiện tại

| Thành phần | Block | Chiều vào/ra | Layers | Heads | FFN |
|---|---|---:|---:|---:|---:|
| Online context | `SBTransformerBlock` | 256 → 256 | 1 | 8 | 1024 |
| Predictor | `SBTransformerBlock` | 128 → 128 | 1 | 8 | 512 |
| Target context | EMA copy của online context | 256 → 256 | 1 | 8 | 1024 |

Cả context và predictor bật positional encoding, dùng pre-norm và dropout 0. Các chiều 256 và 128 đều chia hết cho 8 heads.

Các trọng số/lịch hiện có được giữ nguyên: `jepa_weight=0.1`, `mask_ratio=0.5`, `mask_span=16`, predictor dimension 128, momentum 0.996 → 1.0.

Chỉ cấu hình JEPA của recipe LibriMix SepFormer được chuyển từ Mamba sang Transformer. Những recipe DPMamba khác trong repository không được đổi kiến trúc theo thay đổi này.

## 8. Ý nghĩa học biểu diễn và giới hạn của ý tưởng

Kỳ vọng: ngoài việc decode ra waveform có SI-SNR tốt, latent nguồn còn phải mang thông tin giúp dự đoán biểu diễn nguồn sạch tại các span bị che. Mục tiêu phụ này có thể đóng vai trò regularization cho biểu diễn của encoder và separator.

Tuy nhiên, cần phân biệt kỳ vọng với điều đã được chứng minh:

1. **Mask sau separator, không phải trước separator.** SepFormer đã nhìn thấy toàn bộ mixture khi tạo `H`. Các latent không bị che có thể đã mang thông tin từ vùng sau này bị mask. Vì vậy đây không phải một tác vụ mà mọi đường truy cập thông tin của vùng bị che đều đã bị loại bỏ.
2. Context nhìn thấy zero tại vị trí mask, nhưng các output ở vị trí visible của context vẫn tham gia attention; predictor dùng các output đó để dự đoán target.
3. EMA target và target normalization là các cơ chế hiện có; không nên coi chúng là bằng chứng lý thuyết rằng biểu diễn không thể collapse hoặc rằng chất lượng separation chắc chắn tăng.
4. JEPA Transformer chạy trên **toàn bộ từng đoạn latent hợp lệ**, không dùng cơ chế dual-path/chunk size 250 của separator. Với regular self-attention, chi phí attention tăng bậc hai theo chiều dài đoạn; full-length audio có thể tốn bộ nhớ hơn nhánh Mamba cũ.
5. Khi có padding/gap, từng đoạn hợp lệ được xử lý riêng, không attention xuyên qua gap; positional encoding cũng bắt đầu lại theo từng lần gọi model trên đoạn.
6. Cần thực nghiệm baseline SepFormer và SepFormer+JEPA với điều kiện tương đương để kết luận về hiệu quả; smoke test không thay thế đánh giá SI-SNRi/SDRi trên dữ liệu thật.

## 9. Inference và checkpoint

Inference chỉ cần online encoder, SepFormer MaskNet và decoder. Không cần gọi JEPA hay target branch.

Checkpoint training có lưu toàn bộ module JEPA, gồm online heads, projection, mask token, target copies và bộ đếm EMA. **Checkpoint JEPA Mamba cũ không tương thích nguyên trạng với JEPA Transformer mới.** Việc chuyển sang architecture mới cần một chiến lược khởi tạo/resume rõ ràng, không nên xem như resume nguyên vẹn cùng một experiment.

## 10. Mã nguồn và trạng thái kiểm tra

Các điểm triển khai chính:

- [Cấu hình SepFormer + Transformer JEPA](hparams/LibriMix/sepformer.yaml#L121-L200).
- [PIT alignment và cộng JEPA objective](train_wsj0mix.py#L70-L119).
- [Lấy latent sau mask trước decoder](train_wsj0mix.py#L141-L168).
- [Masked prediction và target branch](modules/jepa.py#L143-L191).
- [Khởi tạo và cập nhật EMA](modules/jepa.py#L193-L216).

Hai repo DPRNN và DPMamba được dùng chỉ để tham khảo cách thêm mục tiêu JEPA phụ và chọn họ sequence model phù hợp với backbone; không thay separator ở repo này bằng các backbone đó, không sửa các repo tham khảo.

Sau khi chuyển cấu hình, đã kiểm tra trên CPU bằng batch giả ngắn:

- Load HyperPyYAML, xác nhận chỉ hai mục `JEPAContext` và `JEPAPredictor` thay đổi.
- Forward riêng context `[2,31,256]` và predictor `[2,31,128]`, giữ đúng shape và output hữu hạn.
- Forward SepFormer → waveform PIT → JEPA, loss hữu hạn.
- Forward với padding/gap và nguồn hoàn toàn không hợp lệ, loss hữu hạn; kiểm tra target đóng băng và cập nhật EMA.

Chưa chạy backward/optimizer, full training, đo chất lượng trên dữ liệu thật hoặc kiểm tra GPU memory với full-length audio.

## Tóm tắt một câu

**Giữ nguyên SepFormer để tách nguồn; khi training, dùng Transformer context/predictor dự đoán EMA embedding của clean source tại những span bị che trên latent nguồn đã PIT-align, rồi cộng loss masked prediction vào loss SI-SNR để học đồng thời.**
