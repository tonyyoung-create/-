# Chat History (摘要與主要對話)

此檔案為你與 AI 助手之間的對話紀錄摘要（含主要問題、修正與指令）。

---

1. 使用者: 要求將原始 `空氣品質預測.py` 改為可部署到 Streamlit Cloud 的應用，保留 CSV 和 API 使用，改用 Streamlit UI 取代 LINE Notify。

2. 助手: 將腳本重構為 Streamlit 應用，新增 `requirements.txt` 與 `README.md`。

3. 使用者: 執行時出現 `ModuleNotFoundError: No module named 'tensorflow'`。

4. 助手: 提供兩個方案：安裝 TensorFlow 或改用 sklearn 備援；使用者選擇使用 sklearn。

5. 助手: 將程式改為 sklearn-only（GradientBoostingRegressor），避免 TensorFlow import 錯誤。並增加模型儲存為 `.pkl`。

6. 使用者: 出現 API SSL 驗證錯誤（CERTIFICATE_VERIFY_FAILED）。

7. 助手: 加入 SSL 例外處理（遇到 SSLError 時 retry verify=False 並停用警告），並說明安全風險與替代方式（certifi 或修正系統憑證）。

8. 使用者: Streamlit 顯示多項污染物被標示為「資料不足」。

9. 助手: 調整前處理，取消對整個 DataFrame 做 `dropna()`，改為保留插值結果，並在每個污染物訓練前單獨做 `dropna()`，以保留更多時間點。

10. 使用者: 要求修正圖表錯誤、把對話紀錄加入並推到 GitHub、並告知要在 Streamlit 上上傳哪些檔案。

11. 助手: 修正圖表（改為實線加單點預測），新增 `chat_history.md`（本檔）、新增 `push_to_github.ps1` 推送腳本，並在本地初始化 git 並 commit（未推到遠端）。

---

完整對話已被濃縮為上述重點流程。若需要「逐句」完整逐條紀錄（非摘要），我可以生成包含每次使用者與助手訊息的完整文本檔，請回覆確認。
