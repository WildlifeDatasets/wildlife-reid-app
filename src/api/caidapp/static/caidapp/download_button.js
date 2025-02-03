function pollTaskStatus(taskId, alertTimeout) {
    console.log("Checking status of the download task: " + taskId + ", alertTimeout=" + alertTimeout);
    $.get(`/caidapp/check_zip_status/${taskId}/`, function(data) {
        console.log("data=" + data);
        if (data.status === "ready") {
            console.log("Download is ready");
            clearTimeout(alertTimeout);
            // Show the download link
            console.log(data);
            console.log(data.download_url);

            // here would be better to use the original download link
            console.log("creating link to download the file from the server. data.download_url=" + data.download_url);
            const downloadUrl = data.download_url;
            const link = document.createElement("a");
            link.href = downloadUrl;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            console.log("Download link clicked.");
        } else if (data.status === "pending") {
            console.log("... still pending, checking again in 3 seconds");
            setTimeout(function() {
                pollTaskStatus(taskId, alertTimeout);
            }, 3000); // Poll every 3 seconds
        } else if (data.status === "error") {
            alert("Error: " + data.message);
        }
    });
}


$(document).on("click", ".download-mediafiles-button", function(event) {
    event.preventDefault(); // Prevent default link behavior

    // Extract the group ID dynamically
    console.log("Starting download of media files in javascript");
    console.log("data-id=")
    console.log( $(this).data("data-id") ) ;
    var uploadedarchiveId;
    var uploadedarchiveUrl;
    if ($(this).data("data-id")) {
        uploadedarchiveId = $(this).data("data-id");
        uploadedarchiveUrl = `/caidapp/download_zip_for_mediafiles/${uploadedarchiveId}/`

    }
    else {
        uploadedarchiveUrl = "/caidapp/download_zip_for_mediafiles/";
    }
    // print the id of the clicked element
    console.log(uploadedarchiveId);
    console.log("url=")
    console.log(uploadedarchiveUrl);

    // Set a timeout to show an additional alert if the download isn't ready after 5 seconds
    const alertTimeout = setTimeout(function() {
        alert("The file is being prepared. The download will start automatically once it's ready.");
    }, 500);
    console.log("Preparing ZIP with media files");
    // Send the request for this specific group
    $.get(uploadedarchiveUrl, function(data) {
        console.log("Calling pollTaskStatus(data), data=");
        console.log(data);
        pollTaskStatus(data.task_id, alertTimeout); // Reuse pollTaskStatus logic
    }).fail(function() {
        clearTimeout(alertTimeout);
        console.error("An error occurred while starting the download process.");
        alert("An error occurred while starting the download process.");
    });
});


