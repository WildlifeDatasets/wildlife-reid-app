function pollTaskStatus(taskId, alertTimeout) {
    $.get(`/caidapp/check_zip_status/${taskId}/`, function(data) {
        if (data.status === "ready") {
            clearTimeout(alertTimeout);
            // Show the download link
            console.log(data);
            console.log(data.download_url);

            const downloadUrl = data.download_url;
            const link = document.createElement("a");
            link.href = downloadUrl;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        } else if (data.status === "pending") {
            setTimeout(function() {
                pollTaskStatus(taskId);
            }, 3000); // Poll every 3 seconds
        } else if (data.status === "error") {
            alert("Error: " + data.message);
        }
    });
}


$(document).on("click", ".download-mediafiles-button", function(event) {
    event.preventDefault(); // Prevent default link behavior

    // Extract the group ID dynamically

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

    // Set a timeout to show an additional alert if the download isn't ready after 2 seconds
    const alertTimeout = setTimeout(function() {
        alert("The file is being prepared. The download will start automatically once it's ready.");
    }, 1000);
    // Send the request for this specific group
    $.get(uploadedarchiveUrl, function(data) {
        console.log(data);
        pollTaskStatus(data.task_id, alertTimeout); // Reuse pollTaskStatus logic
    }).fail(function() {
        clearTimeout(alertTimeout);
        alert("An error occurred while starting the download process.");
    });
});


