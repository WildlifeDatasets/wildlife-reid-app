


# Changelog

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