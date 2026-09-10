/**
 * Mobile Phone Defect Detection - Main Page
 * ==========================================
 * WeChat Mini Program frontend for capturing phone photos, sending them
 * to the detection API, and displaying the grading report.
 *
 * Configure the API endpoint in `utils/config.js` (copy from config.example.js).
 */

const { API_BASE } = require("../../utils/config.js");

Page({
  data: {
    allResults: {},       // { view: { detections, image_name, url } }
    viewImages: {},       // view -> annotated image URL for preview
    currentView: "",
    resultImage: "",
    report: null,
    basePrice: 3000,
    showAlgo: false,
  },

  onPriceChange(e) {
    const price = parseInt(e.detail.value, 10);
    if (!isNaN(price) && price > 0) {
      this.setData({ basePrice: price });
    }
  },

  uploadImage(e) {
    const view = e.currentTarget.dataset.view;
    const that = this;

    wx.showLoading({ title: "AI analyzing..." });

    wx.chooseMedia({
      count: 1,
      success(res) {
        const path = res.tempFiles[0].tempFilePath;
        wx.uploadFile({
          url: `${API_BASE}/api/v1/predict`,
          filePath: path,
          name: "file",
          formData: { view_name: view },
          timeout: 30000,
          success(ret) {
            wx.hideLoading();
            try {
              const resData = JSON.parse(ret.data);
              const fullUrl = resData.result_image_url;
              const fileName = fullUrl.substring(fullUrl.lastIndexOf("/") + 1);

              const dataStore = that.data.allResults;
              dataStore[view] = {
                detections: resData.detections,
                image_name: fileName,
                url: fullUrl,
              };

              const imgs = that.data.viewImages;
              imgs[view] = fullUrl;

              that.setData({
                allResults: dataStore,
                viewImages: imgs,
                resultImage: fullUrl,
                currentView: view,
              });
              wx.showToast({ title: `${view} ready` });
            } catch (err) {
              console.error(err);
              wx.showToast({ title: "Parse failed", icon: "none" });
            }
          },
          fail() {
            wx.hideLoading();
            wx.showToast({ title: "Connection timeout", icon: "none" });
          },
        });
      },
    });
  },

  getFinalGrade() {
    const that = this;

    if (Object.keys(this.data.allResults).length === 0) {
      wx.showToast({ title: "Please capture at least one photo", icon: "none" });
      return;
    }

    wx.showLoading({ title: "Generating report..." });

    wx.request({
      url: `${API_BASE}/api/v1/grade`,
      method: "POST",
      data: {
        all_results: that.data.allResults,
        base_price: that.data.basePrice,
      },
      timeout: 30000,
      header: { "Content-Type": "application/json" },
      success(res) {
        wx.hideLoading();
        if (res.statusCode === 200 && res.data.report) {
          that.setData({ report: res.data.report });
          wx.showToast({ title: "Grading complete", icon: "success" });
        } else {
          wx.showToast({ title: "Grading failed", icon: "none" });
        }
      },
      fail() {
        wx.hideLoading();
        wx.showToast({ title: "Network request failed", icon: "none" });
      },
    });
  },

  previewDefect(e) {
    const item = e.currentTarget.dataset.item;
    const targetUrl = this.data.viewImages[item.pos];

    if (!targetUrl) {
      wx.showToast({ title: "Original image lost", icon: "none" });
      return;
    }

    wx.previewImage({
      current: targetUrl,
      urls: Object.values(this.data.viewImages),
    });
  },

  toggleAlgo() {
    this.setData({ showAlgo: !this.data.showAlgo });
  },
});
