(function () {
    const form = document.getElementById("new-upload-form");
    if (!form) {
        return;
    }

    const dropzone = document.getElementById("new-upload-dropzone");
    const fileInput = document.getElementById("id_upload_files");
    const relativePathsInput = document.getElementById("id_upload_relative_paths");
    const fileSummary = document.getElementById("new-upload-file-summary");
    const pathSummary = document.getElementById("new-upload-path-summary");
    const spreadsheetSummary = document.getElementById("new-upload-spreadsheet-summary");
    const spreadsheetColumns = document.getElementById("new-upload-spreadsheet-columns");
    const spreadsheetButton = document.getElementById("new-upload-spreadsheet-button");
    const directoryButton = document.getElementById("new-upload-directory-button");
    const directoryBadge = document.getElementById("new-upload-directory-badge");
    const directoryStructure = document.getElementById("id_directory_structure");
    const directoryMappingInput = document.getElementById("id_directory_mapping");
    const pathRegexInput = document.getElementById("id_path_regex");
    const spreadsheetColumnMappingInput = document.getElementById("id_spreadsheet_column_mapping");
    const pathBasicPanel = document.getElementById("new-upload-path-basic");
    const pathMapper = document.getElementById("new-upload-path-mapper");
    const advancedRegexInput = document.getElementById("new-upload-path-regex");
    const advancedDirectoryPanel = document.getElementById("upload-directory-advanced");
    const regexChatGptLink = document.getElementById("new-upload-regex-chatgpt-link");
    const columnMapper = document.getElementById("new-upload-column-mapper");
    const progress = document.getElementById("new-upload-progress");
    const progressBar = progress.querySelector(".progress-bar");
    const localityInput = document.getElementById("id_locality_at_upload");
    const checkedAtInput = document.getElementById("id_locality_check_at");
    const uploadTargetInputs = Array.from(document.querySelectorAll('input[name="upload_target"]'));
    const containsIdentitiesInput = document.getElementById("id_contains_identities");
    const identifiedDatasetOption = document.getElementById("new-upload-identified-dataset-option");
    const localityWarning = document.getElementById("new-upload-locality-warning");
    const placeBadge = document.getElementById("new-upload-place-badge");
    const dateBadge = document.getElementById("new-upload-date-badge");
    const targetBadge = document.getElementById("new-upload-target-badge");
    const localities = JSON.parse(form.dataset.localities || "[]");
    let isUploading = false;
    let droppedManifest = [];
    const pathRoles = [
        ["", "Ignore"],
        ["locality", "Locality"],
        ["taxon", "Taxon"],
        ["identity", "Identity"],
        ["check_date", "Check date"],
    ];
    const columnRoles = [
        ["", "Ignore"],
        ["original_path", "Media path"],
        ["taxon", "Taxon"],
        ["unique_name", "Identity"],
        ["locality_name", "Locality"],
        ["datetime", "Datetime"],
        ["latitude", "Latitude"],
        ["longitude", "Longitude"],
    ];
    const roleRegexByName = {
        check_date: "(?P<check_date>\\d{4}-?\\d{2}-?\\d{2})",
        locality: "(?P<locality>[^/]+)",
        taxon: "(?P<taxon>[^/]+)",
        identity: "(?P<identity>[^/]+)",
    };
    const identityLastDirectoryRegex = "^(?:.*/)?(?P<identity>[^/]+)/[^/]+$";

    function getSelectedUploadTarget() {
        if (!uploadTargetInputs.length) {
            return null;
        }
        const checkedRadio = uploadTargetInputs.find((input) => input.type === "radio" && input.checked);
        if (checkedRadio) {
            return checkedRadio.value;
        }
        const hiddenInput = uploadTargetInputs.find((input) => input.type === "hidden");
        return hiddenInput ? hiddenInput.value : null;
    }

    function showPanel(id) {
        const element = document.getElementById(id);
        if (element && window.bootstrap) {
            bootstrap.Collapse.getOrCreateInstance(element, {toggle: false}).show();
        }
    }

    function setSummaryText(element, text) {
        element.textContent = text;
        element.classList.toggle("d-none", !text);
    }

    function preventBrowserDropNavigation(event) {
        event.preventDefault();
        event.stopPropagation();
        if (event.dataTransfer) {
            event.dataTransfer.dropEffect = "copy";
        }
    }

    function fileSuffix(filename) {
        const index = filename.lastIndexOf(".");
        return index >= 0 ? filename.slice(index).toLowerCase() : "";
    }

    function isSpreadsheet(file) {
        return [".csv", ".xls", ".xlsx"].includes(fileSuffix(file.name));
    }

    function isZip(file) {
        return fileSuffix(file.name) === ".zip";
    }

    function isMediaPath(path) {
        return [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"].includes(fileSuffix(path));
    }

    function decodeZipFilename(bytes) {
        return new TextDecoder("utf-8").decode(bytes);
    }

    function findEndOfCentralDirectory(bytes) {
        for (let index = bytes.length - 22; index >= 0; index -= 1) {
            if (
                bytes[index] === 0x50 &&
                bytes[index + 1] === 0x4b &&
                bytes[index + 2] === 0x05 &&
                bytes[index + 3] === 0x06
            ) {
                return index;
            }
        }
        return -1;
    }

    async function listZipFilenames(file) {
        const tailSize = Math.min(file.size, 65557);
        const tailOffset = file.size - tailSize;
        const tailBytes = new Uint8Array(await file.slice(tailOffset).arrayBuffer());
        const eocdIndex = findEndOfCentralDirectory(tailBytes);
        if (eocdIndex < 0) {
            return [];
        }

        const view = new DataView(tailBytes.buffer);
        const centralDirectorySize = view.getUint32(eocdIndex + 12, true);
        const centralDirectoryOffset = view.getUint32(eocdIndex + 16, true);
        const centralDirectoryBytes = new Uint8Array(
            await file.slice(centralDirectoryOffset, centralDirectoryOffset + centralDirectorySize).arrayBuffer()
        );
        const centralView = new DataView(centralDirectoryBytes.buffer);
        const filenames = [];
        let offset = 0;

        while (offset + 46 <= centralDirectoryBytes.length) {
            const signature = centralView.getUint32(offset, true);
            if (signature !== 0x02014b50) {
                break;
            }
            const filenameLength = centralView.getUint16(offset + 28, true);
            const extraLength = centralView.getUint16(offset + 30, true);
            const commentLength = centralView.getUint16(offset + 32, true);
            const filenameStart = offset + 46;
            const filenameEnd = filenameStart + filenameLength;
            filenames.push(decodeZipFilename(centralDirectoryBytes.slice(filenameStart, filenameEnd)));
            offset = filenameEnd + extraLength + commentLength;
        }

        return filenames;
    }

    function parseZipMetadata(filename) {
        if (fileSuffix(filename) !== ".zip") {
            return null;
        }
        const stem = filename.replace(/\.zip$/i, "");
        const datePattern = "(\\d{4}-?\\d{2}-?\\d{2})";
        const first = stem.match(new RegExp(`^${datePattern}[_ ](.+)$`));
        const second = stem.match(new RegExp(`^(.+?)[_ ]${datePattern}$`));
        const match = first || second;
        if (!match) {
            return null;
        }
        const date = first ? match[1] : match[2];
        const locality = first ? match[2] : match[1];
        return {
            locality: locality,
            date: date.length === 8 ? `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}` : date,
        };
    }

    function parseCsvHeader(text) {
        const firstLine = text.split(/\r?\n/)[0] || "";
        return firstLine
            .split(",")
            .map((column) => column.trim().replace(/^"|"$/g, ""))
            .filter(Boolean);
    }

    function guessColumnRole(column) {
        const normalized = column.trim().toLowerCase().replaceAll(" ", "_");
        const aliases = {
            "original_path": "original_path",
            "original_path": "original_path",
            "mediafile": "original_path",
            "media_file": "original_path",
            "taxon": "taxon",
            "category": "taxon",
            "unique_name": "unique_name",
            "identity": "unique_name",
            "locality_name": "locality_name",
            "location_name": "locality_name",
            "datetime": "datetime",
            "date": "datetime",
            "lat": "latitude",
            "latitude": "latitude",
            "lon": "longitude",
            "longitude": "longitude",
        };
        return aliases[normalized] || "";
    }

    function createColumnRoleSelect(column) {
        const select = document.createElement("select");
        select.className = "form-select form-select-sm new-upload-column-role";
        select.dataset.column = column;
        const guessedRole = guessColumnRole(column);
        for (const role of columnRoles) {
            const option = document.createElement("option");
            option.value = role[0];
            option.textContent = role[1];
            if (role[0] === guessedRole) {
                option.selected = true;
            }
            select.appendChild(option);
        }
        select.addEventListener("change", updateColumnMappingFromUi);
        return select;
    }

    function renderColumnMapper(columns) {
        columnMapper.innerHTML = "";
        if (!columns.length) {
            spreadsheetColumnMappingInput.value = "";
            return;
        }
        const wrapper = document.createElement("div");
        wrapper.className = "d-flex flex-wrap gap-2 align-items-end";
        for (const column of columns) {
            const item = document.createElement("div");
            item.className = "border rounded p-2 bg-body-tertiary";
            const label = document.createElement("div");
            label.className = "small text-muted mb-1";
            label.textContent = column;
            item.appendChild(label);
            item.appendChild(createColumnRoleSelect(column));
            wrapper.appendChild(item);
        }
        columnMapper.appendChild(wrapper);
        updateColumnMappingFromUi();
    }

    function updateColumnMappingFromUi() {
        const selects = Array.from(document.querySelectorAll(".new-upload-column-role"));
        const mapping = {};
        for (const select of selects) {
            if (select.value) {
                mapping[select.value] = select.dataset.column;
            }
        }
        spreadsheetColumnMappingInput.value = JSON.stringify(mapping);
    }

    async function updateZipSpreadsheetPreview(zipFiles) {
        const zipSpreadsheetNames = [];
        for (const file of zipFiles) {
            const filenames = await listZipFilenames(file);
            zipSpreadsheetNames.push(
                ...filenames.filter((filename) => [".csv", ".xls", ".xlsx"].includes(fileSuffix(filename)))
            );
        }
        return zipSpreadsheetNames;
    }

    async function listZipMediaPaths(zipFiles) {
        const mediaPaths = [];
        for (const file of zipFiles) {
            const filenames = await listZipFilenames(file);
            mediaPaths.push(...filenames.filter(isMediaPath));
        }
        return mediaPaths;
    }

    async function updateSpreadsheetPreview(spreadsheetFiles, zipFiles) {
        const zipSpreadsheetNames = await updateZipSpreadsheetPreview(zipFiles);
        if (!spreadsheetFiles.length && !zipSpreadsheetNames.length) {
            spreadsheetButton.classList.add("d-none");
            setSummaryText(spreadsheetSummary, "");
            spreadsheetColumns.textContent = "Upload a CSV to preview columns in the browser. XLS/XLSX columns are read on the server during preparation.";
            renderColumnMapper([]);
            return;
        }

        spreadsheetButton.classList.remove("d-none");
        const spreadsheetNames = [
            ...spreadsheetFiles.map((file) => file.name),
            ...zipSpreadsheetNames.map((name) => `${name} inside ZIP`),
        ];
        setSummaryText(spreadsheetSummary, `Spreadsheet detected: ${spreadsheetNames.join(", ")}`);
        const csvFile = spreadsheetFiles.find((file) => fileSuffix(file.name) === ".csv");
        if (!csvFile) {
            spreadsheetColumns.textContent = zipSpreadsheetNames.length
                ? "Spreadsheet file found inside ZIP. Columns will be read on the server during upload preparation."
                : "XLS/XLSX columns will be read on the server during upload preparation.";
            renderColumnMapper([]);
            return;
        }

        const reader = new FileReader();
        reader.onload = function () {
            const columns = parseCsvHeader(String(reader.result || ""));
            spreadsheetColumns.textContent = columns.length
                ? `CSV columns: ${columns.join(", ")}`
                : "CSV header was not detected.";
            renderColumnMapper(columns);
        };
        reader.readAsText(csvFile.slice(0, 4096));
    }

    function updateLocalityWarning() {
        const value = localityInput.value.trim();
        if (value && !localities.includes(value)) {
            localityWarning.textContent = "Unknown locality. A new locality will be created.";
            placeBadge.classList.remove("d-none");
        } else {
            localityWarning.textContent = "";
            placeBadge.classList.add("d-none");
        }
    }

    function updateBadges() {
        if (checkedAtInput.value) {
            dateBadge.classList.remove("d-none");
        } else {
            dateBadge.classList.add("d-none");
        }

        if (targetBadge && uploadTargetInputs.length) {
            const selectedTarget = getSelectedUploadTarget();
            if (selectedTarget === "identification") {
                targetBadge.textContent = "re-id";
                targetBadge.className = "badge text-bg-success";
            } else if (selectedTarget) {
                targetBadge.textContent = "taxa";
                targetBadge.className = "badge text-bg-primary";
            } else {
                targetBadge.textContent = "required";
                targetBadge.className = "badge text-bg-primary";
            }
        }
    }

    function updateUploadTargetUi() {
        if (identifiedDatasetOption) {
            const selectedTarget = getSelectedUploadTarget();
            const identifyUpload = selectedTarget === "identification";
            identifiedDatasetOption.classList.toggle("d-none", !identifyUpload);
            if (!identifyUpload && containsIdentitiesInput) {
                containsIdentitiesInput.checked = false;
            }
        }
        updateBadges();
    }

    function pathWithoutFilename(path) {
        const parts = path.split("/").filter(Boolean);
        return parts.length > 1 ? parts.slice(0, -1) : [];
    }

    function createRoleSelect(index, value) {
        const select = document.createElement("select");
        select.className = "form-select form-select-sm new-upload-path-role";
        select.dataset.index = String(index);
        for (const role of pathRoles) {
            const option = document.createElement("option");
            option.value = role[0];
            option.textContent = role[1];
            if (role[0] === value) {
                option.selected = true;
            }
            select.appendChild(option);
        }
        select.addEventListener("change", updateDirectoryMappingFromUi);
        return select;
    }

    function guessRole(part) {
        if (/^\d{4}-?\d{2}-?\d{2}$/.test(part)) {
            return "check_date";
        }
        return "";
    }

    function renderPathMapper(examplePath) {
        const directoryParts = pathWithoutFilename(examplePath);
        if (!directoryParts.length) {
            pathMapper.innerHTML = '<div class="small text-muted">Upload a directory or files with relative paths to see an example here.</div>';
            directoryStructure.value = "";
            directoryMappingInput.value = "";
            return;
        }

        pathMapper.innerHTML = "";
        const wrapper = document.createElement("div");
        wrapper.className = "d-flex flex-wrap gap-2 align-items-end";

        directoryParts.forEach((part, index) => {
            const item = document.createElement("div");
            item.className = "border rounded p-2 bg-body-tertiary";
            const label = document.createElement("div");
            label.className = "small text-muted mb-1";
            label.textContent = `/${part}`;
            item.appendChild(label);
            item.appendChild(createRoleSelect(index, guessRole(part)));
            wrapper.appendChild(item);
        });

        const filename = examplePath.split("/").filter(Boolean).at(-1);
        const fileItem = document.createElement("div");
        fileItem.className = "border rounded p-2";
        fileItem.innerHTML = `<div class="small text-muted mb-1">/${filename}</div><span class="badge text-bg-secondary">filename</span>`;
        wrapper.appendChild(fileItem);

        pathMapper.appendChild(wrapper);
        const hint = document.createElement("div");
        hint.className = "form-text mt-2";
        hint.textContent = "If metadata is inside filename, use Advanced regex for now.";
        pathMapper.appendChild(hint);
        updateDirectoryMappingFromUi();
        updateRegexChatGptLink(examplePath);
    }

    function updateRegexChatGptLink(examplePath) {
        if (!regexChatGptLink) {
            return;
        }
        const prompt = [
            "Help me write a Python regular expression for parsing a wildlife dataset relative file path.",
            "The regex should use named groups only from: taxon, locality, identity, check_date.",
            "Explain the regex briefly for a non-programmer.",
            "Example path:",
            examplePath || "taxon/identity/Karel__001.jpg",
        ].join("\n");
        regexChatGptLink.href = `https://chatgpt.com/?q=${encodeURIComponent(prompt)}`;
    }

    function buildPathRegexFromUi() {
        const selects = Array.from(document.querySelectorAll(".new-upload-path-role"));
        const rolePositions = {};
        for (const select of selects) {
            if (select.value) {
                rolePositions[select.value] = Number(select.dataset.index);
            }
        }
        const positions = Object.values(rolePositions);
        if (!positions.length) {
            return "";
        }
        const maxPosition = Math.max(...positions);
        const parts = [];
        for (let position = 0; position <= maxPosition; position += 1) {
            const matchingRole = Object.keys(rolePositions).find((role) => rolePositions[role] === position);
            parts.push(roleRegexByName[matchingRole] || "[^/]+");
        }
        return `^${parts.join("/")}/[^/]+$`;
    }

    function applyIdentifiedDatasetPathPreset() {
        const selects = Array.from(document.querySelectorAll(".new-upload-path-role"));
        if (!selects.length || !containsIdentitiesInput || !containsIdentitiesInput.checked) {
            return;
        }

        selects.forEach((select) => {
            select.value = "";
        });
        selects.at(-1).value = "identity";
        updateDirectoryMappingFromUi();
        pathRegexInput.value = identityLastDirectoryRegex;
        if (advancedRegexInput) {
            advancedRegexInput.value = identityLastDirectoryRegex;
        }
        showPanel("upload-directory-panel");
    }

    function updateDirectoryMappingFromUi() {
        const selects = Array.from(document.querySelectorAll(".new-upload-path-role"));
        const mapping = {};
        const structure = [];
        for (const select of selects) {
            const role = select.value;
            const index = Number(select.dataset.index);
            if (role) {
                mapping[role] = index;
                structure.push(`{${role}}`);
            } else {
                structure.push("*");
            }
        }
        directoryStructure.value = structure.filter((part) => part !== "*").length ? structure.join("/") : "";
        directoryMappingInput.value = JSON.stringify(mapping);
        const generatedRegex = buildPathRegexFromUi();
        pathRegexInput.value = generatedRegex;
        if (advancedRegexInput && !advancedDirectoryPanel.classList.contains("show")) {
            advancedRegexInput.value = generatedRegex;
        }
    }

    async function updateFileSummary() {
        const files = Array.from(fileInput.files || []);
        const manifest = droppedManifest.length === files.length
            ? droppedManifest
            : files.map((file, index) => ({
                index: index,
                filename: file.name,
                relative_path: file.webkitRelativePath || file.name,
            }));
        relativePathsInput.value = JSON.stringify(manifest);

        const spreadsheetFiles = files.filter(isSpreadsheet);
        const zipFiles = files.filter(isZip);
        const uploadFiles = files.filter((file) => !isSpreadsheet(file));
        setSummaryText(fileSummary, files.length
            ? `${uploadFiles.length} upload file(s), ${spreadsheetFiles.length} spreadsheet(s).`
            : "");

        const relativePaths = manifest.map((item) => item.relative_path);
        const zipMediaPaths = await listZipMediaPaths(zipFiles);
        const pathsForMapping = [
            ...relativePaths.filter((path) => path.includes("/")),
            ...zipMediaPaths.filter((path) => path.includes("/")),
        ];
        setSummaryText(pathSummary, pathsForMapping.length
            ? `${pathsForMapping.length} media path(s) include directories.`
            : "");

        if (pathsForMapping.length) {
            directoryButton.classList.add("text-primary");
            directoryBadge.classList.remove("d-none");
            renderPathMapper(pathsForMapping[0]);
            applyIdentifiedDatasetPathPreset();
        } else {
            directoryButton.classList.remove("text-primary");
            directoryBadge.classList.add("d-none");
            directoryStructure.value = "";
            directoryMappingInput.value = "";
            renderPathMapper("");
        }

        await updateSpreadsheetPreview(spreadsheetFiles, zipFiles);

        const zipMetadata = files.length === 1 ? parseZipMetadata(files[0].name) : null;
        if (zipMetadata) {
            localityInput.value = zipMetadata.locality;
            checkedAtInput.value = zipMetadata.date;
            showPanel("upload-location-panel");
            showPanel("upload-date-panel");
            updateLocalityWarning();
            updateBadges();
        }
    }

    fileInput.addEventListener("change", function () {
        droppedManifest = [];
        updateFileSummary();
    });
    localityInput.addEventListener("blur", updateLocalityWarning);
    localityInput.addEventListener("input", updateLocalityWarning);
    checkedAtInput.addEventListener("input", updateBadges);
    uploadTargetInputs.forEach((input) => input.addEventListener("change", updateUploadTargetUi));
    if (containsIdentitiesInput) {
        containsIdentitiesInput.addEventListener("change", function () {
            if (containsIdentitiesInput.checked) {
                applyIdentifiedDatasetPathPreset();
            }
        });
    }
    advancedRegexInput.addEventListener("input", function () {
        pathRegexInput.value = advancedRegexInput.value;
    });
    advancedDirectoryPanel.addEventListener("show.bs.collapse", function () {
        pathBasicPanel.classList.add("d-none");
        if (!advancedRegexInput.value) {
            advancedRegexInput.value = pathRegexInput.value || buildPathRegexFromUi();
        }
    });
    advancedDirectoryPanel.addEventListener("hide.bs.collapse", function () {
        pathBasicPanel.classList.remove("d-none");
        updateDirectoryMappingFromUi();
    });

    window.addEventListener("dragover", preventBrowserDropNavigation);
    window.addEventListener("drop", preventBrowserDropNavigation);

    dropzone.addEventListener("dragenter", function (event) {
        preventBrowserDropNavigation(event);
        dropzone.classList.add("border-primary");
    });

    dropzone.addEventListener("dragover", function (event) {
        preventBrowserDropNavigation(event);
        dropzone.classList.add("border-primary");
    });

    dropzone.addEventListener("dragleave", function (event) {
        event.stopPropagation();
        dropzone.classList.remove("border-primary");
    });

    dropzone.addEventListener("drop", function (event) {
        preventBrowserDropNavigation(event);
        dropzone.classList.remove("border-primary");
        handleDroppedItems(event.dataTransfer);
    });

    function readEntryFile(entry) {
        return new Promise((resolve, reject) => {
            entry.file(resolve, reject);
        });
    }

    function readDirectoryEntries(reader) {
        return new Promise((resolve, reject) => {
            reader.readEntries(resolve, reject);
        });
    }

    async function collectEntryFiles(entry, prefix) {
        if (entry.isFile) {
            const file = await readEntryFile(entry);
            return [{
                file: file,
                relative_path: `${prefix}${file.name}`,
            }];
        }

        if (!entry.isDirectory) {
            return [];
        }

        const reader = entry.createReader();
        const collected = [];
        let entries = await readDirectoryEntries(reader);
        while (entries.length) {
            for (const child of entries) {
                collected.push(...await collectEntryFiles(child, `${prefix}${entry.name}/`));
            }
            entries = await readDirectoryEntries(reader);
        }
        return collected;
    }

    async function handleDroppedItems(dataTransfer) {
        const items = Array.from(dataTransfer.items || []);
        const entries = items
            .map((item) => item.webkitGetAsEntry ? item.webkitGetAsEntry() : null)
            .filter(Boolean);
        const hasDirectory = entries.some((entry) => entry.isDirectory);

        if (!hasDirectory) {
            droppedManifest = [];
            fileInput.files = dataTransfer.files;
            await updateFileSummary();
            return;
        }

        const collected = [];
        for (const entry of entries) {
            collected.push(...await collectEntryFiles(entry, ""));
        }

        if (!collected.length) {
            droppedManifest = [];
            fileInput.files = dataTransfer.files;
            await updateFileSummary();
            return;
        }

        const transfer = new DataTransfer();
        droppedManifest = collected.map((item, index) => {
            transfer.items.add(item.file);
            return {
                index: index,
                filename: item.file.name,
                relative_path: item.relative_path,
            };
        });
        fileInput.files = transfer.files;
        await updateFileSummary();
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        const formData = new FormData(form);
        progress.classList.remove("d-none");
        progressBar.style.width = "0%";
        progressBar.textContent = "0%";
        isUploading = true;

        const xhr = new XMLHttpRequest();
        xhr.open("POST", window.location.href);
        xhr.upload.addEventListener("progress", function (progressEvent) {
            if (!progressEvent.lengthComputable) {
                return;
            }
            const percent = Math.round((progressEvent.loaded / progressEvent.total) * 99);
            progressBar.style.width = `${percent}%`;
            progressBar.textContent = `${percent}%`;
        });
        xhr.addEventListener("load", function () {
            isUploading = false;
            progressBar.style.width = "100%";
            progressBar.textContent = "100%";
            let response = {};
            try {
                response = JSON.parse(xhr.responseText || "{}");
            } catch (error) {
                progress.classList.add("d-none");
                return;
            }
            if (response.html) {
                document.getElementById("appbody").innerHTML = response.html;
            }
        });
        xhr.addEventListener("error", function () {
            isUploading = false;
            progress.classList.add("d-none");
        });
        xhr.send(formData);
    });

    window.addEventListener("beforeunload", function (event) {
        if (!isUploading) {
            return;
        }
        event.preventDefault();
        event.returnValue = "An upload is in progress. Are you sure you want to leave?";
    });

    updateFileSummary();
    updateUploadTargetUi();
    updateBadges();
    updateRegexChatGptLink("");
})();
