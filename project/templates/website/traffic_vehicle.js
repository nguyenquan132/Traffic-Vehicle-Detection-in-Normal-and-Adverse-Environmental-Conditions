
const myForm = document.getElementById("uploadForm")
const upload = document.getElementById("fileInput")
const video = document.getElementById("video")

myForm.addEventListener("submit", e => {
    e.preventDefault();
    upload.click();
});

upload.addEventListener("change", async() => {
    const file = upload.files[0];
    // Tạo FormData để gửi tệp
    const formData = new FormData();
    formData.append('video', file);
    
    // Sử dụng fetch API để gửi dữ liệu
    try{
        const response = await fetch("http://127.0.0.1:5000/upload", {
                                    method: "POST",
                                    body: formData,
                                    });

        if (response.ok) {
            const videoUrl = "http://127.0.0.1:5000/video_stream";  
            video.innerHTML = `<img src="${videoUrl}" width="800" alt="Video Stream">`;
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An unexpected error occurred.');
    } 
});