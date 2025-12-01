
///---------------------------------------------------------------------------------------------///
///---------------------------------------------------------------------------------------------///


function showTemporaryAlert(message) {
    const alertElement = document.createElement('div');
    alertElement.textContent = message;
    alertElement.style.position = 'fixed';
    alertElement.style.top = '20px';
    alertElement.style.left = '50%';
    alertElement.style.transform = 'translateX(-50%)';
    alertElement.style.backgroundColor = 'rgba(0,0,0,0.8)';
    alertElement.style.color = 'white';
    alertElement.style.padding = '10px 20px';
    alertElement.style.borderRadius = '5px';
    alertElement.style.zIndex = '1000';

    document.body.appendChild(alertElement);

    setTimeout(() => {
        alertElement.remove();
    }, 2000);
}

// Call this function when the page loads
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
    connectWebSocketcrossing();
    connectAlertWebSocket();
});

//------------------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------------------------------------------------------------------------------------//




function getFirstResultForKIS(){
    console.log(exportedImages)
    return exportedImages[0]
}

function getFirstResultForVQA(){
    const vqa=document.getElementsByClassName("vqa-input") 
    return [exportedImages[0],vqa[0].value]
}



async function submit_to_dres(){
    let frame_info=0
    if (activeTask=="kis"){
    frame_info= getFirstResultForKIS()
    } else{
    frame_info= getFirstResultForVQA()  
    }
    const item=frame_info.frameInfo.split('-')[0];
    const frame= parseInt(frame_info.frameId)*1000;
    const sessionID=localStorage.getItem('sessionId');
    const url= `http://192.168.20.164:5000/api/v1/submit?item=${item}&frame=${frame}&session=${sessionID}`
    const result = await fetch(url, {
    method: "GET",
    })
    .then(response => {
    if (!response.ok) {
        // If the response status is 401 (Unauthorized)
        if (response.status === 401) {
        alert("Unauthorized access. Please check your credentials.");
        } 
        // If the response status is 404 (Not Found)
        else if (response.status === 404) {
        alert("WRONG!!");
        } 
        // Other possible errors
        else {
        alert(`Error: ${response.status}`);
        }
        throw new Error(`HTTP status code ${response.status}`);
    }
    return response.json();
    })
    .then(data => {
    console.log('Success:', data);
    // Process data
    })
    .catch(error => {
    console.error('Error:', error);
    });
}

function submit_to_dres_but_dare_devil(item, frame){
    const url= `http://192.168.20.164:6001/api/v1/submit?item=${item}&frame=${frame}`
    console.log(url)
    fetch(url, {
    method: "GET",
    });
}


async function submit_to_dres_v2() {
    const backendSubmitUrl = "http://localhost:8000/dres/submit";
    let payload = {};

    try {
        if (activeTask === "kis") {
            const frame_info = getFirstResultForKIS();
            if (!frame_info) { alert("Chưa chọn ảnh!"); return; }

            const item = frame_info.frameInfo.split('-')[0];
            // frameId hiện tại đang là giây (float), nhân 1000 và làm tròn thành int
            const ms = Math.round(parseFloat(frame_info.frameId) * 1000); 

            payload = {
                "video_id": item,
                "timestamp_ms": ms, // Gửi số nguyên mili-giây
                "text_answer": null
            };

        } else {
            const frame_info = getFirstResultForVQA();
            if (!frame_info || !frame_info[0]) { alert("Chưa có thông tin VQA!"); return; }

            const item = frame_info[0].frameInfo.split('-')[0];
            const answer_vqa = frame_info[1];
            const ms = Math.round(parseFloat(frame_info[0].frameId) * 1000);

            payload = {
                "video_id": item,
                "timestamp_ms": ms,
                "text_answer": answer_vqa
            };
        }

        console.log("Sending payload:", payload); // Log để kiểm tra

        const response = await fetch(backendSubmitUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        
        if (response.ok) {
            alert("✅ SUBMIT THÀNH CÔNG!\n" + JSON.stringify(data));
        } else {
            console.error(data);
            alert("❌ LỖI TỪ BACKEND: " + (data.detail || JSON.stringify(data)));
        }

    } catch (error) {
        alert("❌ LỖI HỆ THỐNG: " + error.message);
    }
}


async function submitFrameInfo(url, body) {
    try {
        const response = await fetch(url, {
            method: "POST",
            body: JSON.stringify(body)
        });

        if (!response.ok) {
            let message;
            if (response.status === 401) {
                message = "Unauthorized access. Please check your credentials.";
            } else if (response.status === 404) {
                message = "WRONG!!";
            } else {
                message = `Error: ${response.status}`;
            }
            sendAlertViaWebSocket(message);
            throw new Error(`HTTP status code ${response.status}`);
        }

        const data = await response.json();
        if (data.submission === 'WRONG') {
            sendAlertViaWebSocket('Submission wrong!');
        } else {
            console.log('Success:', data);
            sendAlertViaWebSocket('Submission successful!');
            // Process data
        }
    } catch (error) {
        console.error('Error:', error);
        sendAlertViaWebSocket(`Error: ${error.message}`);
    }
}