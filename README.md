# 空氣品質訓練與預測 (Streamlit)

這個專案將原本的 `空氣品質預測.py` 改為一個可以部署於 Streamlit Cloud 的 Web App。功能如下：

- 讀取本地 CSV（`空氣品質小時值_彰化縣_二林站.csv`）並合併 API 的即時資料
- 資料前處理：線性插值（最多補 4 筆），再刪除仍有缺值的時間點
- 可在網頁上選擇污染物並訓練 LSTM 模型（即時訓練）
- 預測下一個時間點的值，顯示圖表與 AQI 分類

部署與執行：

1. 本地測試

   - 建議建立 Python 虛擬環境並安裝套件：

     pip install -r requirements.txt

   - 在專案目錄下啟動 Streamlit：

     streamlit run 空氣品質預測.py

2. 部署到 Streamlit Cloud

   - 在 GitHub 建立儲存庫並推上此專案（包含 `requirements.txt` 與 `空氣品質小時值_彰化縣_二林站.csv`）
   - 到 Streamlit Cloud 建立新的 App，連結 GitHub 儲存庫並部署

注意事項

- Streamlit Cloud 的檔案系統為短期儲存（session-based）。若需長期保存模型，請將模型上傳至雲端儲存（例如 Google Drive、S3）或把訓練過程獨立執行並把模型檔加入 repo。
- 若 CSV 欄位名稱與程式預期不同，請在 CSV 中確認時間欄位、污染物欄位名稱與數值欄位（monitordate、itemengname、concentration）對應。

如需我幫你微調 UI 或增加更多監控/排程功能，我可以繼續修改。