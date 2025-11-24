# Changelog


## 2026-11

* [changed] the media file update use observations now
* [changed] the media files are converted to `.webp` on upload
* [changed] the media file import procedure respect the orientation now
* [added] Upload several media files instead of archive

## 2025-10
* [changed]  DJANGO_SECRET_KEY is now stored in `.env` file

## 2025-09
* [added] Notification on finished identification
* [changed] FGVC subset extracted from FGVC package
* [added] List view for localities and identities
* [changed] Merging is now processed on backgroud
* [added] Separated AnimalObservation from MediaFile in model

## 2025-08

* [added] Suggest representative media file during identification
* [added] Lazy loading of identities in re-identification
* [added] Set representative from media file list
* [added] Pictures in cards in the identity dash
* [added] Suggestions on low number of representative media files
* [added] Status messages of identity suggestion

## 2025-07
* [fixed] Removed ultralytics dependency (by pinning yolo5)
* [added] Add not identified media files to the list (to be identified manually)
* [fixed] List of not confirmed now doesn't skip doubled score
* [fixed] Show remaining identities if one of the suggestion is None
* [changed] Select representative media files in identification according to the orientation
* [added] Automatic postponed run of init identification on changed list of representative media files
* [added] Re-identification init and run automatized
* [changed] Aspect ratio of the media files is fixed

## 2025-06

* [added] Name, Last name, and email in the identity
* [added] Name and Last name shown in impersonation
* [added] Orientation import from CSV
* [added] Coat type import from CSV
* [added] Hide base dataset in the identity
* [changed] Suggest file name for download of media files

## 2025-05

* [added] AI consent
* [added] New identity from media file detail
* [update] Fixed error on some Cuddeback camera trap
* [added] Dash for identity processing

## 2025-05

* [added] Home page
* [added] Identify sorted by score
* [changed] Update by spreadsheet is now more robust
* [changed] If the uploaded archive is uploaded as single taxon, the taxon is used instead of the prediction
* [fixed] Crash on not identified media files view when no locality in one of media files 

## 2025-05

* [fixed] Form behavior fixed for Missing taxon in media file
* [fixed] Verification of media files
* [changed] Most of the choice lists are sorted now
* [added] Localities of Identity
* [added] List of closest localities
* [changed] Simplification of previews, thumbnails and static thumbnails generation
* [added] CSV and XLSX import and export of identities 

## 2025-05

* [fixed] Download ZIP with media files in production
* [changed]  Length of individuality prolonged to 100 characters
* [added] Ordering in Locailities
* [changed] Generic Locality List view used
* [added] Ordering in identities
* [added] Merge multiple identities
* [added] Suggest code extraction from the name of identity 
* [changed] Priority of loading media files in the detail view
* [added] Orientation detection introduced
* [added] Filter of Localitites and Identities
* [added] Area added to Locality
* [changed] Reworked search and filter in media files
* [changed] Category renamed to taxon
* [added] Search for Locality and identity

## 2025-05
* [added] Added carousel for media files in the detail view
* [changed] Logging in taxon worker
* [added] Using `uv` installer for api Docker
* [added] NDOP export
* [added] Merge localities
* [added] Merge localities suggestion
* [added] Merge identities suggestion

## 2024-05

* [added] Merge identity
* [added] Update Media files in upload with spreadsheet file
* [added] Django debug toolbar
* [added] Double carousel in identification to show more images of the identity
* [fix] Media files visible to all if user has no workgroup
* [added] Impersonated staff person can now run classification
* [changed] Disable link to media files before the processing is done
* [changed] EXIFs calculated in one batch
* [changed] Added trace in the map
* [added] Base dataset view added
* [added] Added pygwalker for media files
* [changed] Block identification init if identification is running

## 2024-05

* [added] Extend sequence of media files
* [changed] Order media files in sequence together
* [added] Download media files on background
* [added] Select all media files on the page, or in the selection
* [added] Page title in media files upload

## 2024-05

* [changed] Separated dev and prod
* [added] OCR for date time in media files
* [added] Stats view

* [added] GIF together with video on video detail view
* [added] Sequences in image update
* [added] Filtration of media files
* [added] Support for external synology drive
* [added] Preparation for dev and prod environments

## 2024-05

* [added] Video conversion to mp4
* [added] Better date and location detection in file name during upload
* [added] Next step button in uploaded archives view
* [added] Dates of camera check have now a view
* [changed] Camera trap checks view is reworked
* [added] Status badge
* [changed] renamed uploadedarchive_detail to uploadedarchive_mediafiles

## 2024-05

* [changed] Fix count of files after upload
* [changed] Camera trap check card image is now on the top of the card
* [changed] Login page redesigned: image added, cleared
* [changed] Output files named by convention {locality}_{date}_{original_name}_{taxon}_{identity}
* [fixed] All download XLSX/CSV/Images are now the same
* [added] Download XLSX for uploaded archive added

* [added] Overview taxons in media file view
* [changed] In impersonation order users by name
* [added] New view for processes with taxon classification
* [added] Suggestions for Not Classified
* [added] Show message after upload
* [added] Synology import now supports also "{location} / {date}" format
* [added] Date and location are automatically filled in the upload archive view
* [added] Warning when leaving the upload page before the upload is finished
* [added] Richer ordering possibilities for camera trap checks
* [added] Taxon update form
* [added] Taxon parent

## 2024-05

* [added] Checkbox on overviewed taxon of mediafile
* [added] Checkbox on no missing taxons in camera trap check
* [added] Bulk overview of media files
* [added] Double checkbox on overviewed all media files of camera trap check
* [added] Media file sorting
* [added] Count of media files in one view
* [added] Link to camera trap check from not identified media files
* [changed] Pagination of media files and camera trap checks
* [changed] Links from media file card to location and camera trap check
* [added] Button to got to identity confirmation directly from list of media files
* [added] Impersonate user in admin
* [fixed] Error in taxon in media files view
* [changed] Pagination of media files and camera trap checks
* [added] Overview for single uploaded archive
* [changed] UI for media files is more compact
* [changed] UI for uploaded archives is more compact now
* [added] Private mode

* [added] Reading the date and time from video media files
* [fixed] Remove media files on "Taxon classification (force init)"
* [added] Added support for extra wide monitor on media files view
* [added] Date picker in upload archive view

## 2024-05

* [added] video files are streamed in the detail view
* [added] Download original file for mediafile
* [added] Show mediafile in admin on detail view
* [changed] return back to the previous page on running taxon_worker and identification_worker
* [added] Order Camera Trap Checks
* [added] First and last captured taxon in Camera Trap Check
* [added] Links to location in cards
* [added] Share Identity
* [added] Show media files from location
* [added] Measure GPU memory usage
* [changed] Limit number of worker to 1 in taxon_worker and 1 in identification_worker
* [added] Import locations by CSV or XLSX
* [changed] Locality view reworked
* [added] List of camera trap checks
* [added] Export XLSX for media files
* [added] Show number of media files with taxon for identification

## 2024-05

* [added] Bulk import of media files using Synology Drive
* [added] Error message as status tooltip
* [added] Sex and coat type added to the individual identity
* [added] Orientation added to Media File
* [added] Download CSV file for every list of media files
* [changed] Special view for statistics
* [added] Select taxon for identification
* [changed] YOLOv5 is loaded in Docker to make the first run faster
* [changed] Limit number of workers in taxon_worker and identification_worker
* [changed] The uploads in the identification view are now selected by taxon_for_identification is not None
* [added] Number of representative media files for each identity
* [changed] Unspotted added into the coat type
* [added] Code and juv_code in identity
* [added] Location check date added to Uploaded Archive
* [added] SAM - large model used and downloaded from original url

## 2024-05

* [added] Taxon classification by detection if necessary
* [fixed] Re-run of taxon classification works again for workgroup users
* [added] Paired points from LOFTR are now shown in the detail view
* [changed] gpu:1 used for SAM
* [added] List all representative files for identification
* [added] Download original archive
* [added] Range of dates is visualized for every control upload
* [changed] Increased number of workers in webapp
* [changed] Do the thumbnail generation in parallel

## 2024-05

* [added] Visible name for locations
* [added] Extended edit options for locations
* [added] Taxon statistics on list of mediafiles
* [changed] Layout of the list of mediafiles
* [fixed] Query for media files works now together with filters like taxon, filter, ...

## 2024-05

* [added] Icon showing the mediafile is representative
* [fixed] The single species identified media files are set representative
* [added] Buttons are now organized in workflow scheme
* [added] Expected time of processing is now shown for taxon classification
* [added] Location is stored with precision of 3 decimal places
* [added] Access to the list of all not identified media files from identification view
* [added] Added scheme to species upload
* [added] Sample data support
* [added] Added autocomplete for location at upload

## 2024-05

* [changed] Easier deployment by better setup of migrations and Google Auth
* [fixed] Album thumbnail is now the first image in album
* [fixed] Uploaded archive in now in added to the correct workflow
* [changed] Sidebar for identities is now a dropdown
* [added] Location coordinates by clicking on map
* [fixed] Failure when mediafile datetime is NaN
* [added] On removal of mediafile or archive from database, the files are also removed from the storage

## 2023-05

* [added] Edit in admin button for Uploaded Archive
* [added] View with all not identified media files
