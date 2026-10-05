/* Leaflet presents privacy-safe prepared records; Python owns scope and filtering. */
(() => {
  "use strict";

  const form = document.getElementById("listing-filters");
  const status = document.getElementById("map-status");
  const emptyState = document.getElementById("map-empty-state");
  const emptyTitle = document.getElementById("map-empty-title");
  const emptyDescription = document.getElementById("map-empty-description");
  const fitButton = document.getElementById("fit-results");
  const counts = {
    matched: document.getElementById("matched-count"),
    mapped: document.getElementById("mapped-count"),
    missing: document.getElementById("missing-count"),
  };
  const countFormat = new Intl.NumberFormat("en-US");
  const priceFormat = new Intl.NumberFormat("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 0});
  const numberFormat = new Intl.NumberFormat("en-US", {maximumFractionDigits: 2});
  const peruBounds = [[-18.5, -82], [0.5, -68]];
  let map;
  let activeRequest;
  let currentClusters;
  let selectedMarker;

  const numeric = (value) => typeof value === "number" && Number.isFinite(value);
  const text = (value) => typeof value === "string" && value.trim() ? value : "Unknown";
  const price = (value) => numeric(value) ? `${priceFormat.format(value)} USD` : "Unknown";
  const area = (value) => numeric(value) ? `${numberFormat.format(value)} m²` : "Unknown";
  const rooms = (value) => numeric(value) ? countFormat.format(value) : "Unknown";

  function setStatus(message, error = false) {
    status.textContent = message;
    status.classList.toggle("is-error", error);
  }

  function showEmpty(title, description) {
    emptyTitle.textContent = title;
    emptyDescription.textContent = description;
    emptyState.hidden = false;
  }

  function resetDetails() {
    document.getElementById("details-empty").hidden = false;
    document.getElementById("details-content").hidden = true;
    document.getElementById("detail-fields").replaceChildren();
    selectedMarker = null;
  }

  function paragraph(parent, value, className) {
    const node = document.createElement("p");
    node.textContent = value;
    if (className) node.className = className;
    parent.append(node);
    return node;
  }

  function tooltip(listing) {
    const node = document.createElement("div");
    const heading = document.createElement("strong");
    heading.textContent = price(listing.price_usd);
    node.append(heading);
    paragraph(node, `${text(listing.property_type)} · ${area(listing.surface_total)}`);
    paragraph(node, `Bedrooms: ${rooms(listing.bedrooms)} · Bathrooms: ${rooms(listing.bathrooms)}`);
    paragraph(node, `Region: ${text(listing.l2)} · Locality: ${text(listing.l4)}`, "tooltip-location");
    paragraph(node, "Approximate location", "tooltip-approximate");
    return node;
  }

  function showDetails(listing, marker) {
    if (selectedMarker && selectedMarker.getElement()) selectedMarker.getElement().classList.remove("selected-marker");
    selectedMarker = marker;
    if (marker.getElement()) marker.getElement().classList.add("selected-marker");
    document.getElementById("details-empty").hidden = true;
    document.getElementById("details-content").hidden = false;
    document.getElementById("detail-type").textContent = text(listing.property_type);
    document.getElementById("detail-price").textContent = price(listing.price_usd);
    document.getElementById("detail-location").textContent = [listing.l4, listing.l2].filter((value) => typeof value === "string" && value.trim()).join(" · ") || "Location name unknown";
    const fields = [
      ["Total area", area(listing.surface_total)],
      ["Bedrooms", rooms(listing.bedrooms)],
      ["Bathrooms", rooms(listing.bathrooms)],
      ["Region", text(listing.l2)],
      ["Locality", text(listing.l4)],
      ["Approx. latitude", numeric(listing.lat) ? `${listing.lat.toFixed(2)}°` : "Unknown"],
      ["Approx. longitude", numeric(listing.lon) ? `${listing.lon.toFixed(2)}°` : "Unknown"],
    ];
    const rows = document.createDocumentFragment();
    fields.forEach(([label, value]) => {
      const row = document.createElement("div");
      const name = document.createElement("dt");
      const detail = document.createElement("dd");
      name.textContent = label;
      detail.textContent = value;
      row.append(name, detail);
      rows.append(row);
    });
    document.getElementById("detail-fields").replaceChildren(rows);
  }

  function listingMarker(listing) {
    const category = listing.property_type === "Casa" ? "casa" : "departamento";
    const marker = L.marker([listing.lat, listing.lon], {
      // The icon contains only fixed decorative markup. Record fields use DOM text.
      icon: L.divIcon({className: `listing-marker ${category}`, html: "<span></span>", iconSize: [15, 15], iconAnchor: [7, 7]}),
      title: `${text(listing.property_type)} · ${price(listing.price_usd)} · approximate location`,
      alt: "Approximate residential listing location",
      keyboard: true,
      riseOnHover: true,
    });
    marker.bindTooltip(() => tooltip(listing), {direction: "top", offset: [0, -10], opacity: 1, className: "listing-tooltip"});
    marker.on("click", () => showDetails(listing, marker));
    return marker;
  }

  function clusterIcon(cluster) {
    const count = cluster.getChildCount();
    const size = count >= 1000 ? 49 : count >= 100 ? 43 : 37;
    const label = document.createElement("span");
    label.textContent = countFormat.format(count);
    return L.divIcon({
      html: label,
      className: `listing-cluster ${count >= 1000 ? "cluster-large" : count >= 100 ? "cluster-medium" : "cluster-small"}`,
      iconSize: [size, size],
    });
  }

  function fitResults() {
    if (!currentClusters) return;
    const bounds = currentClusters.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, {padding: [35, 35], maxZoom: 12, animate: false});
  }

  async function addMarkers(group, markers, controller, alreadyLoaded, total) {
    // Keep each batch below markercluster's timer boundary. This lets a changed
    // filter cancel between batches without a stale plugin timer using a map
    // layer that has already been removed.
    for (let start = 0; start < markers.length; start += 200) {
      if (controller.signal.aborted) return;
      group.addLayers(markers.slice(start, start + 200));
      const processed = Math.min(start + 200, markers.length);
      setStatus(`Loading map points: ${countFormat.format(alreadyLoaded + processed)} of ${countFormat.format(total)}…`);
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }
  }

  async function applyFilters(query) {
    if (!map) return;
    if (activeRequest) activeRequest.abort();
    const controller = new AbortController();
    activeRequest = controller;
    if (currentClusters) map.removeLayer(currentClusters);
    const group = L.markerClusterGroup({chunkedLoading: true, chunkInterval: 80, chunkDelay: 20, maxClusterRadius: 48, showCoverageOnHover: false, spiderfyOnMaxZoom: true, spiderfyDistanceMultiplier: 1.5, iconCreateFunction: clusterIcon});
    currentClusters = group;
    map.addLayer(group);
    resetDetails();
    fitButton.disabled = true;
    emptyState.hidden = true;
    Object.values(counts).forEach((node) => { node.textContent = "—"; });
    setStatus("Loading matching listings…");
    let loaded = 0;
    let lastData = null;
    let offset = 0;
    try {
      do {
        const pageQuery = new URLSearchParams(query);
        pageQuery.set("limit", "2000");
        pageQuery.set("offset", String(offset));
        const response = await fetch(`/api/listings?${pageQuery}`, {signal: controller.signal, headers: {Accept: "application/json"}});
        const data = await response.json();
        if (!response.ok) {
          const error = new Error(typeof data.error === "string" ? data.error : "The listing data could not be loaded.");
          error.httpStatus = response.status;
          throw error;
        }
        if (controller.signal.aborted) return;
        if (!Array.isArray(data.listings)) throw new Error("The listing data could not be read. Please refresh the page.");
        counts.matched.textContent = countFormat.format(data.matched_listings);
        counts.mapped.textContent = countFormat.format(data.mappable_listings);
        counts.missing.textContent = countFormat.format(data.missing_coordinates);
        const markers = data.listings.map(listingMarker);
        await addMarkers(group, markers, controller, loaded, data.mappable_listings);
        if (controller.signal.aborted) return;
        loaded += markers.length;
        lastData = data;
        fitButton.disabled = loaded === 0;
        if (data.next_offset !== null && (!Number.isInteger(data.next_offset) || data.next_offset <= offset)) throw new Error("The listing response is incomplete. Please refresh the page.");
        offset = data.next_offset;
      } while (offset !== null);

      if (controller.signal.aborted) return;
      if (loaded !== lastData.mappable_listings) throw new Error("The listing response is incomplete. Please apply the filters again.");
      if (loaded) {
        fitResults();
        setStatus(`Showing all ${countFormat.format(loaded)} map locations from ${countFormat.format(lastData.matched_listings)} matching listings.`);
      } else if (lastData.matched_listings) {
        setStatus(`${countFormat.format(lastData.matched_listings)} listings match; none have valid map coordinates.`);
        showEmpty("No coordinates for these listings", "The matching records remain part of the data insights. Try another filter to explore mapped listings.");
      } else {
        setStatus("No listings match these filters.");
        showEmpty("No matching listings", "Try a wider price or area range, or reset the filters.");
      }
    } catch (error) {
      if (error.name === "AbortError" || controller.signal.aborted) return;
      // Do not leave a partial response on screen as though it were complete.
      map.removeLayer(group);
      fitButton.disabled = true;
      Object.values(counts).forEach((node) => { node.textContent = "—"; });
      setStatus(error.message || "The listing data could not be loaded. Please try again.", true);
      if (error.httpStatus === 400) {
        showEmpty("Check the filters", "Correct the filter values above, or reset the filters to explore all available listings.");
      } else if (error.httpStatus === 503) {
        showEmpty("Exploration data is not prepared", "After restoring authorized source data, run python -m real_estate.prepare_explorer, then apply the filters again. See the README for setup.");
      } else {
        showEmpty("Listing data unavailable", "Please check the local application connection, then apply the filters again.");
      }
    }
  }

  async function start() {
    if (!window.L || !L.markerClusterGroup || !window.ExplorerFilters) {
      setStatus("The map could not start. Refresh the page after checking the local installation.", true);
      showEmpty("Map unavailable", "The local map assets could not be loaded.");
      return;
    }
    map = L.map("listing-map", {preferCanvas: true, zoomControl: true, minZoom: 3, maxZoom: 18, scrollWheelZoom: true});
    map.fitBounds(peruBounds, {padding: [12, 12], animate: false});
    const tiles = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    });
    let tileFailure = false;
    tiles.on("tileerror", () => {
      tileFailure = true;
      document.getElementById("basemap-status").hidden = false;
    });
    tiles.on("loading", () => { tileFailure = false; });
    tiles.on("load", () => { if (!tileFailure) document.getElementById("basemap-status").hidden = true; });
    tiles.addTo(map);
    fitButton.addEventListener("click", fitResults);
    document.getElementById("peru-view").addEventListener("click", () => map.fitBounds(peruBounds, {padding: [12, 12]}));
    window.ExplorerFilters.bind(form, applyFilters);
    window.ExplorerFilters.loadOptions(form);
    await applyFilters(window.ExplorerFilters.params(form));
  }

  start();
})();
