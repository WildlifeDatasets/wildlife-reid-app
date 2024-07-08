


# Changelog

## 07-2024

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

## 06-2024

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

## 05-2024

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

## 04-2024

* [added] Taxon classification by detection if necessary
* [fixed] Re-run of taxon classification works again for workgroup users
* [added] Paired points from LOFTR are now shown in the detail view
* [changed] gpu:1 used for SAM
* [added] List all representative files for identification
* [added] Download original archive
* [added] Range of dates is visualized for every control upload
* [changed] Increased number of workers in webapp
* [changed] Do the thumbnail generation in parallel

## 03-2024

* [added] Visible name for locations
* [added] Extended edit options for locations
* [added] Taxon statistics on list of mediafiles
* [changed] Layout of the list of mediafiles
* [fixed] Query for media files works now together with filters like taxon, filter, ...

## 02-2024

* [added] Icon showing the mediafile is representative
* [fixed] The single species identified media files are set representative
* [added] Buttons are now organized in workflow scheme
* [added] Expected time of processing is now shown for taxon classification
* [added] Location is stored with precision of 3 decimal places
* [added] Access to the list of all not identified media files from identification view
* [added] Added scheme to species upload
* [added] Sample data support
* [added] Added autocomplete for location at upload

## 01-2024

* [changed] Easier deployment by better setup of migrations and Google Auth
* [fixed] Album thumbnail is now the first image in album
* [fixed] Uploaded archive in now in added to the correct workflow
* [changed] Sidebar for identities is now a dropdown
* [added] Location coordinates by clicking on map
* [fixed] Failure when mediafile datetime is NaN
* [added] On removal of mediafile or archive from database, the files are also removed from the storage

## 12-2023

* [added] Edit in admin button for Uploaded Archive
* [added] View with all not identified media files