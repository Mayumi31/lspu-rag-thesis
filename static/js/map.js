(() => {
  const mapSvg = document.getElementById("campusMap");
  if (!mapSvg) return;

  const mapPanel = document.getElementById("campusMapPanel");
  const mapWorld = document.getElementById("campusMapWorld");
  const mapStatus = document.getElementById("mapStatus");
  const mapImage = mapWorld.querySelector("image");
  const viewWidth = 1440;
  const viewHeight = 810;
  const minZoom = 0.8;
  const maxZoom = 16;
  const zoomStep = 1.25;
  const state = {
    zoom: 1,
    x: 0,
    y: 0,
    pointers: new Map(),
    panGesture: null,
    pinchGesture: null
  };

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function clampPosition() {
    if (state.zoom <= 1) {
      state.x = viewWidth * (1 - state.zoom) / 2;
      state.y = viewHeight * (1 - state.zoom) / 2;
      return;
    }

    state.x = clamp(state.x, viewWidth * (1 - state.zoom), 0);
    state.y = clamp(state.y, viewHeight * (1 - state.zoom), 0);
  }

  function renderMap() {
    clampPosition();
    mapWorld.setAttribute("transform", `translate(${state.x} ${state.y}) scale(${state.zoom})`);
    document.getElementById("mapZoomIn").disabled = state.zoom >= maxZoom;
    document.getElementById("mapZoomOut").disabled = state.zoom <= minZoom;
  }

  function pointFromClient(clientX, clientY) {
    const point = mapSvg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    return point.matrixTransform(mapSvg.getScreenCTM().inverse());
  }

  function zoomAt(nextZoom, point) {
    const boundedZoom = clamp(nextZoom, minZoom, maxZoom);
    const worldX = (point.x - state.x) / state.zoom;
    const worldY = (point.y - state.y) / state.zoom;
    state.zoom = boundedZoom;
    state.x = point.x - worldX * boundedZoom;
    state.y = point.y - worldY * boundedZoom;
    renderMap();
  }

  function resetMap() {
    state.zoom = 1;
    state.x = 0;
    state.y = 0;
    mapStatus.textContent = "Map view reset.";
    renderMap();
  }

  function distanceBetween(first, second) {
    return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
  }

  function midpoint(first, second) {
    return {
      clientX: (first.clientX + second.clientX) / 2,
      clientY: (first.clientY + second.clientY) / 2
    };
  }

  function beginPan(point) {
    state.panGesture = { point, x: state.x, y: state.y };
  }

  function beginPinch() {
    const points = [...state.pointers.values()].slice(0, 2);
    const center = midpoint(points[0], points[1]);
    const centerInMap = pointFromClient(center.clientX, center.clientY);
    state.pinchGesture = {
      distance: distanceBetween(points[0], points[1]),
      zoom: state.zoom,
      worldX: (centerInMap.x - state.x) / state.zoom,
      worldY: (centerInMap.y - state.y) / state.zoom
    };
    state.panGesture = null;
  }

  function handlePointerDown(event) {
    event.preventDefault();
    mapSvg.setPointerCapture(event.pointerId);
    state.pointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });

    if (state.pointers.size === 1) {
      beginPan(pointFromClient(event.clientX, event.clientY));
    } else if (state.pointers.size === 2) {
      beginPinch();
    }
  }

  function handlePointerMove(event) {
    if (!state.pointers.has(event.pointerId)) return;
    event.preventDefault();
    state.pointers.set(event.pointerId, { clientX: event.clientX, clientY: event.clientY });

    if (state.pointers.size >= 2 && state.pinchGesture) {
      const points = [...state.pointers.values()].slice(0, 2);
      const center = midpoint(points[0], points[1]);
      const centerInMap = pointFromClient(center.clientX, center.clientY);
      state.zoom = clamp(state.pinchGesture.zoom * distanceBetween(points[0], points[1]) / state.pinchGesture.distance, minZoom, maxZoom);
      state.x = centerInMap.x - state.pinchGesture.worldX * state.zoom;
      state.y = centerInMap.y - state.pinchGesture.worldY * state.zoom;
      renderMap();
      return;
    }

    if (state.pointers.size === 1 && state.panGesture) {
      const point = pointFromClient(event.clientX, event.clientY);
      state.x = state.panGesture.x + point.x - state.panGesture.point.x;
      state.y = state.panGesture.y + point.y - state.panGesture.point.y;
      renderMap();
    }
  }

  function handlePointerEnd(event) {
    state.pointers.delete(event.pointerId);
    state.pinchGesture = null;
    state.panGesture = null;
    if (state.pointers.size === 1) {
      const remaining = [...state.pointers.values()][0];
      beginPan(pointFromClient(remaining.clientX, remaining.clientY));
    }
  }

  function bindControls() {
    document.getElementById("mapZoomIn").addEventListener("click", () => {
      zoomAt(state.zoom * zoomStep, { x: viewWidth / 2, y: viewHeight / 2 });
    });
    document.getElementById("mapZoomOut").addEventListener("click", () => {
      zoomAt(state.zoom / zoomStep, { x: viewWidth / 2, y: viewHeight / 2 });
    });
    document.getElementById("mapReset").addEventListener("click", resetMap);
    document.getElementById("mapFullscreen").addEventListener("click", async () => {
      try {
        if (document.fullscreenElement === mapPanel) {
          await document.exitFullscreen();
        } else if (mapPanel.requestFullscreen) {
          await mapPanel.requestFullscreen();
        } else {
          mapStatus.textContent = "Fullscreen is not available in this browser.";
        }
      } catch {
        mapStatus.textContent = "Fullscreen could not be opened.";
      }
    });

    document.addEventListener("fullscreenchange", () => {
      const isFullscreen = document.fullscreenElement === mapPanel;
      const button = document.getElementById("mapFullscreen");
      button.setAttribute("aria-label", isFullscreen ? "Exit fullscreen" : "Enter fullscreen");
      button.title = isFullscreen ? "Exit fullscreen" : "Fullscreen";
    });

    mapSvg.addEventListener("pointerdown", handlePointerDown);
    mapSvg.addEventListener("pointermove", handlePointerMove);
    mapSvg.addEventListener("pointerup", handlePointerEnd);
    mapSvg.addEventListener("pointercancel", handlePointerEnd);
    mapSvg.addEventListener("wheel", event => {
      event.preventDefault();
      const point = pointFromClient(event.clientX, event.clientY);
      zoomAt(state.zoom * (event.deltaY < 0 ? zoomStep : 1 / zoomStep), point);
    }, { passive: false });
  }

  bindControls();
  renderMap();

  mapImage.addEventListener("error", () => {
    mapStatus.textContent = "The campus map could not be loaded.";
  });
})();