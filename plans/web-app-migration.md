# Plan: Convert Salamander Re-ID PyQt6 App to Web Application

Convert the existing PyQt6 desktop application for Salamander Re-Identification into a modern, aesthetic web application using FastAPI (Backend) and React (Frontend).

## 1. Architectural Changes

### Backend (FastAPI)
- **Engine/Model persistence**: The `DataManager` and `EmbeddingEngine` will be maintained as singleton-like instances in the FastAPI app state.
- **Endpoints**:
    - `POST /api/metadata`: Upload/load metadata file.
    - `POST /api/config/image-root`: Set the base directory for images.
    - `POST /api/embeddings`: Trigger background embedding generation.
    - `GET /api/embeddings/status`: Poll for progress.
    - `GET /api/identities`: Fetch list of unique identities with metadata.
    - `GET /api/identities/{identity_id}/images`: Fetch all images for a specific ID.
    - `GET /api/validation/{index}`: Get query image and its N-nearest neighbors.
    - `PATCH /api/images/{index}/identity`: Update the Identity ID for a specific record.
    - `POST /api/tasks/detect-errors`: Run error detection logic.
    - `POST /api/tasks/split-data`: Run train/test split.
    - `POST /api/tasks/validation-test`: Run closest set validation test.
    - `GET /api/export`: Export and download the corrected CSV.
    - `GET /api/images/serve/{path:path}`: Static file serving for images (secured/restricted to project root).

### Frontend (React + TypeScript)
- **State Management**: React Context or simple state for data, filters, and current view.
- **Components**:
    - `Sidebar`: Navigation and action buttons (Load, Embed, Split, Detect, Export).
    - `IdentitiesGrid`: Responsive grid of identity cards.
    - `IdentityDetail`: View showing all images of a specific identity.
    - `ValidationView`: Focused view for comparing an image against its closest matches and updating its ID.
    - `ErrorList`: View for browsing detected potential errors.
    - `ImageCard`: Reusable component for displaying salamander images with metadata.
- **Styling**: Modern CSS with Flexbox/Grid, transitions, and a professional "dark-ish" or "clean scientific" theme.

## 2. Implementation Steps

### Phase 1: Backend Foundation
1.  Initialize FastAPI project structure.
2.  Port `data_model.py` and `engine.py` logic to be compatible with FastAPI (handling paths, dataframes).
3.  Implement basic endpoints: metadata loading and image serving.

### Phase 2: Core API Features
1.  Implement embedding generation as a background task.
2.  Implement identity listing and image filtering.
3.  Implement nearest neighbor search and validation data retrieval.
4.  Implement ID update and export functionality.

### Phase 3: Frontend Development
1.  Set up React project with TypeScript.
2.  Create the layout and sidebar.
3.  Build the `IdentitiesGrid` and `IdentityDetail` views.
4.  Build the `ValidationView` (the most critical interactive part).
5.  Add error/test result listings.

### Phase 4: Aesthetics & UX
1.  Apply professional styling: clean shadows, subtle gradients, rounded corners.
2.  Add loading states and progress bars for long-running operations.
3.  Ensure responsiveness for different screen sizes (though primarily targeting desktop use).

## 3. Verification & Testing
- **Data Integrity**: Verify that updating an ID in the web UI correctly updates the underlying DataFrame and reflects in the exported CSV.
- **Performance**: Ensure embedding generation doesn't block the UI and progress is accurately reported.
- **Visuals**: Review the UI for a "modern" feel compared to the PyQt6 version.

## 4. Migration strategy
- Keep the existing `app.py` for reference.
- New code will be in a new directory or alongside (e.g., `web_server.py` and `frontend/`).
