
function checkStatuses(fetch_url) {
    fetch(fetch_url)  // upravte URL dle Vašeho pojmenování
        .then(response => response.json())
        .then(data => {
            // data.archives = pole archivů {id, status}
            let archives = data.archives;

            // 1) Projít si archivy a updatovat DOM elementy přímo (pokud chcete částečný reload)
            archives.forEach(item => {
                let element = document.getElementById("status-" + item.id);
                if (element) {
                    if (element.textContent === item.status ){
                        console.log("Status not changed");
                    }
                    else {
                        console.log("Status changed");
                        console.log(item)
                        element.textContent = item.status;
                        element.className = "badge badge-status rounded-pill bg-" + item.status_style;
                        element.title = item.status_message;
                    }
                }
            });

            // Polling opakovat za 5 vteřin (libovolně nastavit)
            setTimeout(checkStatuses, 5000, fetch_url);
        })
        .catch(err => {
            console.error(err);
            // I v případě chyby to po chvíli zkusit znovu
            setTimeout(checkStatuses, 10000, fetch_url);
        });
}

