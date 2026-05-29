(function () {
    const form = document.getElementById("new-upload-form");
    if (!form) {
        return;
    }

    const dropzone = document.getElementById("new-upload-dropzone");
    const fileInput = document.getElementById("id_upload_files");
    const relativePathsInput = document.getElementById("id_upload_relative_paths");
    const fileSummary = document.getElementById("new-upload-file-summary");
    const selectedFilesButton = document.getElementById("new-upload-selected-files-button");
    const selectedFilesBadge = document.getElementById("new-upload-selected-files-badge");
    const selectedFilesPanel = document.getElementById("new-upload-selected-files-panel");
    const selectedFilesContainer = document.getElementById("new-upload-selected-files");
    const wizardSteps = Array.from(document.querySelectorAll(".upload-wizard-step"));
    const prevStepButton = document.getElementById("new-upload-prev");
    const nextStepButton = document.getElementById("new-upload-next");
    const submitButton = document.getElementById("new-upload-submit");
    const stepLabel = document.getElementById("new-upload-step-label");
    const pathSummary = document.getElementById("new-upload-path-summary");
    const spreadsheetSummary = document.getElementById("new-upload-spreadsheet-summary");
    const metadataCard = document.getElementById("new-upload-metadata-card");
    const metadataEmptyState = document.getElementById("new-upload-metadata-empty-state");
    const metadataContent = document.getElementById("new-upload-metadata-content");
    const spreadsheetColumns = document.getElementById("new-upload-spreadsheet-columns");
    const spreadsheetPathCheck = document.getElementById("new-upload-spreadsheet-path-check");
    const spreadsheetButton = document.getElementById("new-upload-spreadsheet-button");
    const pathAdjustmentButton = document.getElementById("new-upload-path-adjustment-button");
    const pathAdjustmentBadge = document.getElementById("new-upload-path-adjustment-badge");
    const pathAdjustmentPanel = document.getElementById("new-upload-path-adjustment-panel");
    const columnMapperButton = document.getElementById("new-upload-column-mapper-button");
    const columnMapperBadge = document.getElementById("new-upload-column-mapper-badge");
    const columnMapperPanel = document.getElementById("new-upload-column-mapper-panel");
    const pathHintMedia = document.getElementById("new-upload-path-hint-media");
    const pathHintSpreadsheet = document.getElementById("new-upload-path-hint-spreadsheet");
    const pathHintCommon = document.getElementById("new-upload-path-hint-common");
    const pathHintSuggestion = document.getElementById("new-upload-path-hint-suggestion");
    const applyPathAdjustmentButton = document.getElementById("new-upload-apply-path-adjustment");
    const clearPathAdjustmentButton = document.getElementById("new-upload-clear-path-adjustment");
    const directoryButton = document.getElementById("new-upload-directory-button");
    const directoryBadge = document.getElementById("new-upload-directory-badge");
    const directoryStructure = document.getElementById("id_directory_structure");
    const directoryMappingInput = document.getElementById("id_directory_mapping");
    const pathRegexInput = document.getElementById("id_path_regex");
    const spreadsheetColumnMappingInput = document.getElementById("id_spreadsheet_column_mapping");
    const spreadsheetPathAdjustmentInput = document.getElementById("id_spreadsheet_path_adjustment");
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
    let currentStep = 0;
    let selectedEntries = [];
    let currentMediaPaths = [];
    let currentSpreadsheetPreview = null;
    let pathAdjustmentConfig = {remove_prefix: "", add_prefix: ""};
    let pendingPathAdjustmentConfig = {remove_prefix: "", add_prefix: ""};
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
        ["code", "Identity code"],
        ["juv_code", "Juv. code"],
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
    const normalizedSpreadsheetFilename = "mediafile.post_update.csv";

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
        if (!element) {
            return;
        }
        if (window.bootstrap) {
            bootstrap.Collapse.getOrCreateInstance(element, {toggle: false}).show();
        } else {
            element.classList.add("show");
        }
    }

    function setSummaryText(element, text) {
        element.textContent = text;
        element.classList.toggle("d-none", !text);
    }

    function updateWizardUi() {
        wizardSteps.forEach((step, index) => {
            step.classList.toggle("is-active", index === currentStep);
        });
        if (prevStepButton) {
            prevStepButton.disabled = currentStep === 0;
        }
        if (nextStepButton) {
            nextStepButton.classList.toggle("d-none", currentStep === wizardSteps.length - 1);
        }
        if (submitButton) {
            submitButton.classList.toggle("d-none", currentStep !== wizardSteps.length - 1);
        }
        if (stepLabel) {
            stepLabel.textContent = `Step ${currentStep + 1} of ${wizardSteps.length}`;
        }
        window.scrollTo({top: 0, behavior: "smooth"});
    }

    function goToStep(stepIndex) {
        currentStep = Math.max(0, Math.min(stepIndex, wizardSteps.length - 1));
        updateWizardUi();
    }

    function buildSelectedEntryKey(file, relativePath) {
        return [
            relativePath || file.name,
            file.name,
            file.size,
            file.lastModified,
        ].join("::");
    }

    function createSelectedEntry(file, relativePath) {
        const resolvedRelativePath = relativePath || file.webkitRelativePath || file.name;
        return {
            key: buildSelectedEntryKey(file, resolvedRelativePath),
            file: file,
            filename: file.name,
            relative_path: resolvedRelativePath,
        };
    }

    function syncSelectedEntriesToInput() {
        const transfer = new DataTransfer();
        for (const entry of selectedEntries) {
            transfer.items.add(entry.file);
        }
        fileInput.files = transfer.files;
    }

    function renderSelectedFiles() {
        if (!selectedFilesContainer) {
            return;
        }
        selectedFilesContainer.replaceChildren();
        if (selectedFilesButton) {
            selectedFilesButton.classList.toggle("d-none", selectedEntries.length === 0);
            selectedFilesButton.setAttribute("aria-expanded", "false");
        }
        if (selectedFilesBadge) {
            selectedFilesBadge.textContent = String(selectedEntries.length);
        }
        selectedFilesContainer.classList.toggle("d-none", selectedEntries.length === 0);
        if (!selectedEntries.length) {
            if (selectedFilesPanel && window.bootstrap) {
                bootstrap.Collapse.getOrCreateInstance(selectedFilesPanel, {toggle: false}).hide();
            } else if (selectedFilesPanel) {
                selectedFilesPanel.classList.remove("show");
            }
            return;
        }

        const list = document.createElement("div");
        list.className = "d-flex flex-column gap-2";
        for (const entry of selectedEntries) {
            const row = document.createElement("div");
            row.className = "d-flex align-items-start justify-content-between gap-2 border rounded p-2 bg-body-tertiary";

            const info = document.createElement("div");
            info.className = "small";

            const title = document.createElement("div");
            title.className = "fw-semibold";
            title.textContent = entry.filename;
            info.appendChild(title);

            if (entry.relative_path && entry.relative_path !== entry.filename) {
                const detail = document.createElement("div");
                detail.className = "text-muted";
                detail.textContent = entry.relative_path;
                info.appendChild(detail);
            }

            const removeButton = document.createElement("button");
            removeButton.type = "button";
            removeButton.className = "btn-close";
            removeButton.setAttribute("aria-label", `Remove ${entry.filename}`);
            removeButton.addEventListener("click", async function () {
                selectedEntries = selectedEntries.filter((item) => item.key !== entry.key);
                syncSelectedEntriesToInput();
                renderSelectedFiles();
                await updateFileSummary();
            });

            row.appendChild(info);
            row.appendChild(removeButton);
            list.appendChild(row);
        }
        selectedFilesContainer.appendChild(list);
    }

    function mergeSelectedEntries(entries) {
        const seenKeys = new Set(selectedEntries.map((entry) => entry.key));
        for (const entry of entries) {
            if (seenKeys.has(entry.key)) {
                continue;
            }
            seenKeys.add(entry.key);
            selectedEntries.push(entry);
        }
        syncSelectedEntriesToInput();
        renderSelectedFiles();
    }

    function updateMetadataAvailability(hasFiles) {
        if (!metadataCard || !metadataEmptyState || !metadataContent) {
            return;
        }
        metadataCard.classList.toggle("upload-metadata-card-muted", !hasFiles);
        metadataCard.classList.toggle("upload-metadata-card-active", hasFiles);
        metadataEmptyState.classList.toggle("d-none", hasFiles);
        metadataContent.classList.toggle("d-none", !hasFiles);
    }

    function setInfoText(element, text, className) {
        element.textContent = text;
        element.className = `small mt-2 ${className || ""}`.trim();
        element.classList.toggle("d-none", !text);
    }

    function setInfoLines(element, lines, className) {
        const normalizedLines = Array.isArray(lines)
            ? lines.filter((line) => Boolean(line))
            : (lines ? [lines] : []);
        element.replaceChildren();
        element.className = `small mt-2 ${className || ""}`.trim();
        element.classList.toggle("d-none", normalizedLines.length === 0);
        for (const line of normalizedLines) {
            const row = document.createElement("div");
            row.textContent = line;
            element.appendChild(row);
        }
    }

    function setPanelExpanded(panel, button, badge, expanded, expandedLabel, collapsedLabel, expandedBadgeClass, collapsedBadgeClass) {
        if (!panel) {
            return;
        }
        if (window.bootstrap) {
            bootstrap.Collapse.getOrCreateInstance(panel, {toggle: false})[expanded ? "show" : "hide"]();
        } else {
            panel.classList.toggle("show", expanded);
        }
        if (button) {
            button.setAttribute("aria-expanded", expanded ? "true" : "false");
        }
        if (badge) {
            badge.textContent = expanded ? expandedLabel : collapsedLabel;
            badge.className = expanded ? expandedBadgeClass : collapsedBadgeClass;
        }
    }

    function setPathAdjustmentExpanded(expanded) {
        setPanelExpanded(
            pathAdjustmentPanel,
            pathAdjustmentButton,
            pathAdjustmentBadge,
            expanded,
            "open",
            "review",
            "badge text-bg-primary",
            "badge text-bg-warning",
        );
    }

    function setColumnMapperExpanded(expanded) {
        setPanelExpanded(
            columnMapperPanel,
            columnMapperButton,
            columnMapperBadge,
            expanded,
            "open",
            "hidden",
            "badge text-bg-primary",
            "badge text-bg-secondary",
        );
    }

    function updatePathAdjustmentHint(example) {
        if (!pathAdjustmentButton || !pathHintMedia || !pathHintSpreadsheet || !pathHintCommon || !pathHintSuggestion) {
            return;
        }
        if (!example) {
            pathAdjustmentButton.classList.add("d-none");
            setPathAdjustmentExpanded(false);
            pathHintMedia.textContent = "";
            pathHintSpreadsheet.textContent = "";
            pathHintCommon.textContent = "";
            pathHintSuggestion.textContent = "";
            pendingPathAdjustmentConfig = {remove_prefix: "", add_prefix: ""};
            if (applyPathAdjustmentButton) {
                applyPathAdjustmentButton.classList.add("d-none");
            }
            if (clearPathAdjustmentButton) {
                clearPathAdjustmentButton.classList.toggle(
                    "d-none",
                    !pathAdjustmentConfig.remove_prefix && !pathAdjustmentConfig.add_prefix,
                );
            }
            return;
        }

        pathAdjustmentButton.classList.remove("d-none");
        pathHintMedia.textContent = example.mediaPath;
        pathHintSpreadsheet.textContent = example.spreadsheetPath;
        pathHintCommon.textContent = example.commonSuffix;
        pathHintSuggestion.textContent = example.suggestion;
        pendingPathAdjustmentConfig = {
            remove_prefix: example.removePrefix || "",
            add_prefix: example.addPrefix || "",
        };
        const pendingMatchesApplied =
            pendingPathAdjustmentConfig.remove_prefix === pathAdjustmentConfig.remove_prefix
            && pendingPathAdjustmentConfig.add_prefix === pathAdjustmentConfig.add_prefix;
        if (applyPathAdjustmentButton) {
            applyPathAdjustmentButton.classList.toggle("d-none", pendingMatchesApplied);
        }
        if (clearPathAdjustmentButton) {
            clearPathAdjustmentButton.classList.toggle(
                "d-none",
                !pathAdjustmentConfig.remove_prefix && !pathAdjustmentConfig.add_prefix,
            );
        }
        setPathAdjustmentExpanded(true);
    }

    function syncSpreadsheetPathAdjustmentInput() {
        if (spreadsheetPathAdjustmentInput) {
            spreadsheetPathAdjustmentInput.value = JSON.stringify(pathAdjustmentConfig);
        }
        if (clearPathAdjustmentButton) {
            clearPathAdjustmentButton.classList.toggle(
                "d-none",
                !pathAdjustmentConfig.remove_prefix && !pathAdjustmentConfig.add_prefix,
            );
        }
    }

    function adjustSpreadsheetPath(path) {
        let adjustedPath = String(path || "");
        const removePrefix = String(pathAdjustmentConfig.remove_prefix || "").replaceAll("\\", "/").trim();
        const addPrefix = String(pathAdjustmentConfig.add_prefix || "").replaceAll("\\", "/").trim();
        if (removePrefix) {
            const normalizedRemovePrefix = removePrefix.replace(/^\/+|\/+$/g, "");
            if (adjustedPath === normalizedRemovePrefix) {
                adjustedPath = "";
            } else if (adjustedPath.startsWith(`${normalizedRemovePrefix}/`)) {
                adjustedPath = adjustedPath.slice(normalizedRemovePrefix.length + 1);
            }
        }
        if (addPrefix) {
            const normalizedAddPrefix = addPrefix.replace(/^\/+|\/+$/g, "");
            adjustedPath = adjustedPath ? `${normalizedAddPrefix}/${adjustedPath}` : normalizedAddPrefix;
        }
        return adjustedPath;
    }

    function findSharedSuffixExample(mediaPaths, spreadsheetPaths) {
        for (const spreadsheetPath of spreadsheetPaths) {
            for (const mediaPath of mediaPaths) {
                if (mediaPath === spreadsheetPath) {
                    continue;
                }
                if (mediaPath.endsWith(`/${spreadsheetPath}`)) {
                    const prefixToAdd = mediaPath.slice(0, -(spreadsheetPath.length + 1));
                    return {
                        mediaPath: mediaPath,
                        spreadsheetPath: spreadsheetPath,
                        commonSuffix: spreadsheetPath,
                        removePrefix: "",
                        addPrefix: prefixToAdd ? `${prefixToAdd}/` : "",
                        suggestion: prefixToAdd
                            ? `Try prefixing spreadsheet paths with "${prefixToAdd}/".`
                            : "Spreadsheet paths likely need an additional leading directory.",
                    };
                }
                if (spreadsheetPath.endsWith(`/${mediaPath}`)) {
                    const prefixToRemove = spreadsheetPath.slice(0, -(mediaPath.length + 1));
                    return {
                        mediaPath: mediaPath,
                        spreadsheetPath: spreadsheetPath,
                        commonSuffix: mediaPath,
                        removePrefix: prefixToRemove ? `${prefixToRemove}/` : "",
                        addPrefix: "",
                        suggestion: prefixToRemove
                            ? `Try removing the leading prefix "${prefixToRemove}/" from spreadsheet paths.`
                            : "Spreadsheet paths likely contain an extra leading directory.",
                    };
                }
            }
        }
        return null;
    }

    function renderSpreadsheetPathCheck(summary) {
        spreadsheetPathCheck.replaceChildren();
        spreadsheetPathCheck.className = "small mt-2";
        spreadsheetPathCheck.classList.toggle("d-none", !summary || !summary.lines.length);
        if (!summary || !summary.lines.length) {
            return;
        }

        for (const line of summary.lines) {
            const row = document.createElement("div");
            row.className = line.className || "";

            if (line.icon) {
                const icon = document.createElement("i");
                icon.className = `${line.icon} me-1`;
                row.appendChild(icon);
            }

            const text = document.createElement("span");
            text.textContent = line.text;
            row.appendChild(text);

            if (line.actionLabel && line.actionHandler) {
                const button = document.createElement("button");
                button.type = "button";
                button.className = "btn btn-link btn-sm p-0 ms-2 align-baseline";
                button.textContent = line.actionLabel;
                button.addEventListener("click", line.actionHandler);
                row.appendChild(button);
            }

            spreadsheetPathCheck.appendChild(row);
        }
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
        return [".csv", ".xlsx"].includes(fileSuffix(file.name));
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

    function parseXlsxHeader(arrayBuffer) {
        if (!window.XLSX) {
            return [];
        }
        const workbook = window.XLSX.read(arrayBuffer, {type: "array"});
        const firstSheetName = workbook.SheetNames[0];
        if (!firstSheetName) {
            return [];
        }
        const worksheet = workbook.Sheets[firstSheetName];
        const rows = window.XLSX.utils.sheet_to_json(worksheet, {header: 1, raw: false});
        const firstRow = rows[0] || [];
        return firstRow.map((column) => String(column).trim()).filter(Boolean);
    }

    async function readSpreadsheetColumnsFromFile(file) {
        const suffix = fileSuffix(file.name);
        if (suffix === ".csv") {
            const text = await file.text();
            const columns = parseCsvHeader(text);
            const rows = columns.length ? text.split(/\r?\n/).slice(1).filter(Boolean).map((line) => {
                const values = line.split(",").map((value) => value.trim().replace(/^"|"$/g, ""));
                return Object.fromEntries(columns.map((column, index) => [column, values[index] || ""]));
            }) : [];
            return {columns: columns, rows: rows};
        }
        if (suffix === ".xlsx") {
            const buffer = await file.arrayBuffer();
            const columns = parseXlsxHeader(buffer);
            const workbook = window.XLSX.read(buffer, {type: "array"});
            const firstSheetName = workbook.SheetNames[0];
            const worksheet = firstSheetName ? workbook.Sheets[firstSheetName] : null;
            const rows = worksheet ? window.XLSX.utils.sheet_to_json(worksheet, {defval: ""}) : [];
            return {columns: columns, rows: rows};
        }
        return {columns: [], rows: []};
    }

    function describeSpreadsheetColumns(columns, spreadsheetLabel) {
        if (!columns.length) {
            return `${spreadsheetLabel} was detected, but no header row was recognized.`;
        }
        return `${spreadsheetLabel} was detected and is ready for mapping.`;
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
            "code": "code",
            "juv_code": "juv_code",
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
            renderSpreadsheetPathCheck(null);
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(false);
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
        updateSpreadsheetPathCheck(mapping);
    }

    function normalizePathForCompare(path) {
        return String(path || "")
            .replaceAll("\\", "/")
            .replace(/^\.?\//, "")
            .replace(/^\/+|\/+$/g, "")
            .trim()
            .toLowerCase();
    }

    function updateSpreadsheetPathCheck(mapping) {
        if (!currentSpreadsheetPreview || !currentSpreadsheetPreview.rows.length || !currentMediaPaths.length) {
            renderSpreadsheetPathCheck(null);
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(false);
            return;
        }

        const originalPathColumn = mapping.original_path;
        if (!originalPathColumn) {
            renderSpreadsheetPathCheck({
                lines: [
                    {
                        text: "Assign a spreadsheet column to Media path to verify whether spreadsheet paths match uploaded files.",
                        className: "text-muted",
                    },
                ],
            });
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(true);
            return;
        }

        const spreadsheetPaths = currentSpreadsheetPreview.rows
            .map((row) => normalizePathForCompare(adjustSpreadsheetPath(row[originalPathColumn])))
            .filter(Boolean);

        if (!spreadsheetPaths.length) {
            renderSpreadsheetPathCheck({
                lines: [
                    {
                        text: `Column "${originalPathColumn}" is mapped to Media path, but it does not contain any readable paths.`,
                        className: "upload-pathcheck-warning",
                    },
                ],
            });
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(true);
            return;
        }

        const uniqueSpreadsheetPaths = [...new Set(spreadsheetPaths)];
        const uniqueMediaPaths = [...new Set(currentMediaPaths.map(normalizePathForCompare).filter(Boolean))];
        const mediaPathSet = new Set(uniqueMediaPaths);
        const spreadsheetPathSet = new Set(uniqueSpreadsheetPaths);

        let matchedMediaFiles = 0;
        for (const mediaPath of uniqueMediaPaths) {
            if (spreadsheetPathSet.has(mediaPath)) {
                matchedMediaFiles += 1;
            }
        }

        let extraSpreadsheetPaths = 0;
        let suffixOnlyHints = 0;
        for (const spreadsheetPath of uniqueSpreadsheetPaths) {
            if (mediaPathSet.has(spreadsheetPath)) {
                continue;
            }
            const suffixMatch = uniqueMediaPaths.some(
                (mediaPath) => mediaPath.endsWith(`/${spreadsheetPath}`) || spreadsheetPath.endsWith(`/${mediaPath}`),
            );
            if (suffixMatch) {
                suffixOnlyHints += 1;
            } else {
                extraSpreadsheetPaths += 1;
            }
        }

        const missingMediaFiles = uniqueMediaPaths.length - matchedMediaFiles;
        const suffixExample = suffixOnlyHints > 0
            ? findSharedSuffixExample(uniqueMediaPaths, uniqueSpreadsheetPaths)
            : null;

        const lines = [];
        if (matchedMediaFiles === uniqueMediaPaths.length && uniqueMediaPaths.length > 0) {
            lines.push({
                text: `Matched: ${matchedMediaFiles}/${uniqueMediaPaths.length} media files`,
                className: "upload-pathcheck-success fw-semibold",
                icon: "bi bi-check-circle-fill",
            });
        } else if (matchedMediaFiles === 0) {
            lines.push({
                text: `Matched: ${matchedMediaFiles}/${uniqueMediaPaths.length} media files`,
                className: "upload-pathcheck-danger fw-semibold",
            });
        } else {
            lines.push({
                text: `Matched: ${matchedMediaFiles}/${uniqueMediaPaths.length} media files`,
                className: "upload-pathcheck-warning fw-semibold",
            });
        }
        if (missingMediaFiles > 0) {
            lines.push({
                text: `Missing in spreadsheet: ${missingMediaFiles}`,
                className: "upload-pathcheck-warning",
            });
        }
        if (extraSpreadsheetPaths > 0) {
            lines.push({
                text: `Extra in spreadsheet: ${extraSpreadsheetPaths}`,
                className: "upload-pathcheck-warning",
            });
        }
        if (suffixOnlyHints > 0) {
            lines.push({
                text: "Some unmatched paths look similar except for leading directories.",
                className: "upload-pathcheck-warning",
            });
        }
        renderSpreadsheetPathCheck({lines: lines});
        updatePathAdjustmentHint(suffixExample);
        const allMatched = matchedMediaFiles === uniqueMediaPaths.length && uniqueMediaPaths.length > 0;
        setColumnMapperExpanded(!allMatched);
        if (allMatched) {
            setPathAdjustmentExpanded(false);
        }
    }

    async function updateZipSpreadsheetPreview(zipFiles) {
        const zipSpreadsheetNames = [];
        for (const file of zipFiles) {
            const filenames = await listZipFilenames(file);
            zipSpreadsheetNames.push(
                ...filenames.filter((filename) => [".csv", ".xlsx"].includes(fileSuffix(filename)))
            );
        }
        return zipSpreadsheetNames;
    }

    async function readSpreadsheetColumnsFromZipFile(zipFile) {
        if (!window.JSZip) {
            return {filename: "", columns: [], rows: []};
        }
        const zip = await window.JSZip.loadAsync(zipFile);
        const spreadsheetEntries = Object.values(zip.files)
            .filter((entry) => !entry.dir && [".csv", ".xlsx"].includes(fileSuffix(entry.name)));

        if (!spreadsheetEntries.length) {
            return {filename: "", columns: [], rows: []};
        }

        spreadsheetEntries.sort((left, right) => {
            const leftName = left.name.split("/").pop();
            const rightName = right.name.split("/").pop();
            if (leftName === normalizedSpreadsheetFilename && rightName !== normalizedSpreadsheetFilename) {
                return -1;
            }
            if (rightName === normalizedSpreadsheetFilename && leftName !== normalizedSpreadsheetFilename) {
                return 1;
            }
            if (fileSuffix(left.name) === ".csv" && fileSuffix(right.name) === ".xlsx") {
                return -1;
            }
            if (fileSuffix(left.name) === ".xlsx" && fileSuffix(right.name) === ".csv") {
                return 1;
            }
            return left.name.localeCompare(right.name);
        });

        const entry = spreadsheetEntries[0];
        if (fileSuffix(entry.name) === ".csv") {
            const text = await entry.async("text");
            const columns = parseCsvHeader(text);
            return {
                filename: entry.name,
                columns: columns,
                rows: columns.length ? text.split(/\r?\n/).slice(1).filter(Boolean).map((line) => {
                    const values = line.split(",").map((value) => value.trim().replace(/^"|"$/g, ""));
                    return Object.fromEntries(columns.map((column, index) => [column, values[index] || ""]));
                }) : [],
            };
        }

        const arrayBuffer = await entry.async("arraybuffer");
        const columns = parseXlsxHeader(arrayBuffer);
        const workbook = window.XLSX.read(arrayBuffer, {type: "array"});
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = firstSheetName ? workbook.Sheets[firstSheetName] : null;
        return {
            filename: entry.name,
            columns: columns,
            rows: worksheet ? window.XLSX.utils.sheet_to_json(worksheet, {defval: ""}) : [],
        };
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
            spreadsheetColumns.textContent = "Upload a CSV or XLSX, or include one in ZIP, to preview columns and map them to expected fields.";
            currentSpreadsheetPreview = null;
            renderColumnMapper([]);
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(false);
            return;
        }

        spreadsheetButton.classList.remove("d-none");
        spreadsheetButton.setAttribute("aria-expanded", "true");
        showPanel("upload-spreadsheet-panel");
        const spreadsheetNames = [
            ...spreadsheetFiles.map((file) => file.name),
            ...zipSpreadsheetNames.map((name) => `${name} inside ZIP`),
        ];
        setSummaryText(spreadsheetSummary, `Spreadsheet detected: ${spreadsheetNames.join(", ")}`);
        const directSpreadsheet = spreadsheetFiles[0] || null;
        const zipSpreadsheet = directSpreadsheet ? null : (zipFiles[0] || null);
        try {
            if (directSpreadsheet) {
                currentSpreadsheetPreview = await readSpreadsheetColumnsFromFile(directSpreadsheet);
                spreadsheetColumns.textContent = describeSpreadsheetColumns(
                    currentSpreadsheetPreview.columns,
                    directSpreadsheet.name,
                );
                renderColumnMapper(currentSpreadsheetPreview.columns);
                return;
            }
            if (zipSpreadsheet) {
                const zipSpreadsheetPreview = await readSpreadsheetColumnsFromZipFile(zipSpreadsheet);
                if (zipSpreadsheetPreview.filename) {
                    currentSpreadsheetPreview = zipSpreadsheetPreview;
                    spreadsheetColumns.textContent =
                        `${describeSpreadsheetColumns(currentSpreadsheetPreview.columns, `${zipSpreadsheetPreview.filename} inside ZIP`)} ` +
                        "Expected fields such as media path, identity, taxon, locality, and datetime can be mapped below.";
                    renderColumnMapper(currentSpreadsheetPreview.columns);
                    return;
                }
            }
        } catch (error) {
            console.warn("Spreadsheet preview failed", error);
            spreadsheetColumns.textContent = "Spreadsheet detected, but column preview failed in the browser. Try reselecting the file or using CSV.";
            currentSpreadsheetPreview = null;
            renderColumnMapper([]);
            updatePathAdjustmentHint(null);
            setColumnMapperExpanded(false);
            return;
        }

        currentSpreadsheetPreview = null;
        spreadsheetColumns.textContent = zipSpreadsheetNames.length
            ? "Spreadsheet found in ZIP. Expected fields such as media path, identity, taxon, locality, and datetime can be mapped after upload preparation."
            : "Spreadsheet detected, but preview is unavailable. You can still upload it and continue with mapping.";
        renderColumnMapper([]);
        updatePathAdjustmentHint(null);
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
        const files = selectedEntries.map((entry) => entry.file);
        updateMetadataAvailability(files.length > 0);
        const manifest = selectedEntries.map((entry, index) => ({
            index: index,
            filename: entry.filename,
            relative_path: entry.relative_path,
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
        currentMediaPaths = [
            ...relativePaths.filter(isMediaPath),
            ...zipMediaPaths,
        ];
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

    fileInput.addEventListener("change", async function () {
        const addedEntries = Array.from(fileInput.files || []).map((file) => createSelectedEntry(file));
        mergeSelectedEntries(addedEntries);
        await updateFileSummary();
    });
    if (prevStepButton) {
        prevStepButton.addEventListener("click", function () {
            goToStep(currentStep - 1);
        });
    }
    if (nextStepButton) {
        nextStepButton.addEventListener("click", function () {
            goToStep(currentStep + 1);
        });
    }
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
    if (applyPathAdjustmentButton) {
        applyPathAdjustmentButton.addEventListener("click", function () {
            pathAdjustmentConfig = {...pendingPathAdjustmentConfig};
            syncSpreadsheetPathAdjustmentInput();
            updateColumnMappingFromUi();
        });
    }
    if (clearPathAdjustmentButton) {
        clearPathAdjustmentButton.addEventListener("click", function () {
            pathAdjustmentConfig = {remove_prefix: "", add_prefix: ""};
            syncSpreadsheetPathAdjustmentInput();
            updateColumnMappingFromUi();
        });
    }
    if (spreadsheetPathAdjustmentInput) {
        spreadsheetPathAdjustmentInput.value = JSON.stringify(pathAdjustmentConfig);
    }
    updateMetadataAvailability((fileInput.files || []).length > 0);
    updateWizardUi();

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
            mergeSelectedEntries(
                Array.from(dataTransfer.files || []).map((file) => createSelectedEntry(file))
            );
            await updateFileSummary();
            return;
        }

        const collected = [];
        for (const entry of entries) {
            collected.push(...await collectEntryFiles(entry, ""));
        }

        if (!collected.length) {
            mergeSelectedEntries(
                Array.from(dataTransfer.files || []).map((file) => createSelectedEntry(file))
            );
            await updateFileSummary();
            return;
        }

        mergeSelectedEntries(
            collected.map((item) => createSelectedEntry(item.file, item.relative_path))
        );
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
