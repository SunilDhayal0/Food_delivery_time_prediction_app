/**
 * DeliveryTime-X - Frontend Client Logic
 */

// Use the backend origin when the static frontend runs separately.
const isLocalFrontend = ['127.0.0.1', 'localhost'].includes(window.location.hostname);
const API_BASE_URL = isLocalFrontend && window.location.port !== '8010'
  ? 'http://127.0.0.1:8010'
  : window.location.origin;

// Global state
let map = null;
let routeLayer = null;
let pickupMarker = null;
let dropMarker = null;

// DOM Elements
const backendStatusEl = document.getElementById('backend-status');
const formEl = document.getElementById('prediction-form');
const btnSubmit = document.getElementById('btn-submit');
const btnText = btnSubmit.querySelector('.btn-text');
const spinner = btnSubmit.querySelector('.spinner');

const resultCard = document.getElementById('result-card');
const placeholderCard = document.getElementById('placeholder-card');
const errorAlert = document.getElementById('error-alert');
const errorTitle = document.getElementById('error-title');
const errorMsg = document.getElementById('error-msg');

const predMinutesEl = document.getElementById('pred-minutes');
const trafficBadge = document.getElementById('traffic-badge');
const trafficBadgeText = document.getElementById('traffic-badge-text');

const fallbackBox = document.getElementById('fallback-notification');
const fallbackText = document.getElementById('fallback-text');

// Breakdown fields
const bWeather = document.getElementById('b-weather');
const bTraffic = document.getElementById('b-traffic');
const bDistance = document.getElementById('b-distance');
const bCity = document.getElementById('b-city');
const bFestival = document.getElementById('b-festival');

// Autocomplete elements
const fromInput = document.getElementById('from-address');
const toInput = document.getElementById('to-address');
const fromSuggestions = document.getElementById('from-suggestions');
const toSuggestions = document.getElementById('to-suggestions');
const pickupLatInput = document.getElementById('pickup-lat');
const pickupLonInput = document.getElementById('pickup-lon');
const dropLatInput = document.getElementById('drop-lat');
const dropLonInput = document.getElementById('drop-lon');

// Vehicle and other inputs
const vehicleTypeSelect = document.getElementById('vehicle-type');
const vehicleConditionSelect = document.getElementById('vehicle-condition');
const typeOfOrderSelect = document.getElementById('type-of-order');
const multipleDeliveriesSelect = document.getElementById('multiple-deliveries');

// Preset buttons
const presetBtns = document.querySelectorAll('.btn-preset');

// ==========================================
// 1. BACKEND HEALTH CHECK
// ==========================================
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      if (data.model_loaded) {
        backendStatusEl.textContent = 'ML Model Active';
        document.querySelector('.status-dot').style.backgroundColor = '#10b981';
      } else {
        backendStatusEl.textContent = 'Model Not Loaded';
        document.querySelector('.status-dot').style.backgroundColor = '#f59e0b';
      }
    } else {
      throw new Error(`Status ${res.status}`);
    }
  } catch (err) {
    backendStatusEl.textContent = 'Backend Offline';
    document.querySelector('.status-dot').style.backgroundColor = '#ef4444';
  }
}

// ==========================================
// 2. AUTOCOMPLETE WITH DEBOUNCE
// ==========================================
function debounce(fn, delayMs = 350) {
  let timeout;
  return function (...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn.apply(this, args), delayMs);
  };
}

async function fetchAutocomplete(query) {
  if (!query || query.trim().length < 3) return [];
  try {
    const res = await fetch(`${API_BASE_URL}/api/autocomplete?q=${encodeURIComponent(query)}`);
    if (!res.ok) return [];
    return await res.json();
  } catch (err) {
    console.warn('Autocomplete fetch failed:', err);
    return [];
  }
}

function setupAutocomplete(inputEl, suggestionsBox, latHiddenEl, lonHiddenEl) {
  const handler = debounce(async () => {
    const query = inputEl.value.trim();
    // Clear previously stored lat/lon if user is typing a new query
    latHiddenEl.value = '';
    lonHiddenEl.value = '';

    if (query.length < 3) {
      suggestionsBox.innerHTML = '';
      suggestionsBox.classList.add('hidden');
      return;
    }

    const items = await fetchAutocomplete(query);
    suggestionsBox.innerHTML = '';

    if (!items || items.length === 0) {
      suggestionsBox.classList.add('hidden');
      return;
    }

    items.forEach((item) => {
      const div = document.createElement('div');
      div.className = 'suggestion-item';
      div.textContent = item.display_name;
      div.addEventListener('click', () => {
        inputEl.value = item.display_name;
        latHiddenEl.value = item.lat;
        lonHiddenEl.value = item.lon;
        suggestionsBox.innerHTML = '';
        suggestionsBox.classList.add('hidden');
      });
      suggestionsBox.appendChild(div);
    });

    suggestionsBox.classList.remove('hidden');
  }, 350);

  inputEl.addEventListener('input', handler);

  // Hide suggestion list on click outside
  document.addEventListener('click', (e) => {
    if (!inputEl.contains(e.target) && !suggestionsBox.contains(e.target)) {
      suggestionsBox.classList.add('hidden');
    }
  });
}

// ==========================================
// 3. LEAFLET MAP INITIALIZATION & ROUTE RENDER
// ==========================================
function initOrUpdateMap(pickupCoords, dropCoords, geometry) {
  const mapContainer = document.getElementById('route-map');
  if (!mapContainer) return;

  if (!map) {
    map = L.map('route-map', {
      zoomControl: true,
      scrollWheelZoom: false
    });

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
  }

  // Clear existing layers
  if (routeLayer) map.removeLayer(routeLayer);
  if (pickupMarker) map.removeLayer(pickupMarker);
  if (dropMarker) map.removeLayer(dropMarker);

  const pLat = pickupCoords.lat;
  const pLon = pickupCoords.lon;
  const dLat = dropCoords.lat;
  const dLon = dropCoords.lon;

  pickupMarker = L.marker([pLat, pLon]).addTo(map).bindPopup('<b>Pickup (Restaurant)</b>');
  dropMarker = L.marker([dLat, dLon]).addTo(map).bindPopup('<b>Drop (Customer)</b>');

  if (geometry && geometry.coordinates && geometry.coordinates.length > 0) {
    // GeoJSON coordinates are [lon, lat]
    const latLngs = geometry.coordinates.map(coord => [coord[1], coord[0]]);
    routeLayer = L.polyline(latLngs, {
      color: '#ff6b35',
      weight: 5,
      opacity: 0.85,
      dashArray: null,
      lineCap: 'round'
    }).addTo(map);
    map.fitBounds(routeLayer.getBounds(), { padding: [40, 40] });
  } else {
    // Fallback to straight line between pickup and drop
    const straightLine = [[pLat, pLon], [dLat, dLon]];
    routeLayer = L.polyline(straightLine, {
      color: '#ff6b35',
      weight: 4,
      dashArray: '8, 8'
    }).addTo(map);
    map.fitBounds(straightLine, { padding: [40, 40] });
  }

  setTimeout(() => {
    map.invalidateSize();
  }, 200);
}

// ==========================================
// 4. PREDICTION FORM SUBMISSION
// ==========================================
async function handleFormSubmit(e) {
  if (e) e.preventDefault();

  const fromAddr = fromInput.value.trim();
  const toAddr = toInput.value.trim();

  if (!fromAddr || !toAddr) {
    showError('Missing Locations', 'Please provide both pickup and drop addresses.');
    return;
  }

  // Build payload strictly adhering to 6 fields + optional cached coords
  const payload = {
    from_address: fromAddr,
    to_address: toAddr,
    vehicle_type: vehicleTypeSelect.value,
    vehicle_condition: parseInt(vehicleConditionSelect.value, 10),
    type_of_order: typeOfOrderSelect.value,
    multiple_deliveries: parseInt(multipleDeliveriesSelect.value, 10),
    pickup_lat: pickupLatInput.value ? parseFloat(pickupLatInput.value) : null,
    pickup_lon: pickupLonInput.value ? parseFloat(pickupLonInput.value) : null,
    drop_lat: dropLatInput.value ? parseFloat(dropLatInput.value) : null,
    drop_lon: dropLonInput.value ? parseFloat(dropLonInput.value) : null,
  };

  // UI loading state
  setLoading(true);
  hideError();

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Prediction failed with HTTP ${res.status}`);
    }

    renderResults(data);
  } catch (err) {
    showError('Prediction Failed', err.message || 'Unable to compute delivery time.');
  } finally {
    setLoading(false);
  }
}

// ==========================================
// 5. RENDER RESULTS
// ==========================================
function renderResults(data) {
  placeholderCard.classList.add('hidden');
  resultCard.classList.remove('hidden');

  // Predicted minutes
  const mins = data.predicted_minutes !== undefined ? data.predicted_minutes : (data.predicted_time_minutes || '--');
  predMinutesEl.textContent = typeof mins === 'number' ? Math.round(mins) : mins;

  // Traffic badge styling
  const traffic = (data.traffic_level || (data.breakdown && data.breakdown.traffic_density) || 'medium').toLowerCase();
  trafficBadgeText.textContent = `${traffic.toUpperCase()} TRAFFIC`;
  trafficBadge.className = 'pred-badge';
  if (traffic.includes('jam')) {
    trafficBadge.classList.add('badge-jam');
  } else if (traffic.includes('high')) {
    trafficBadge.classList.add('badge-high');
  } else if (traffic.includes('low')) {
    trafficBadge.classList.add('badge-low');
  } else {
    trafficBadge.classList.add('badge-medium');
  }

  // Feature breakdown grid
  const breakdown = data.breakdown || {};
  bWeather.textContent = data.weather || breakdown.weather_condition || 'Sunny';
  bTraffic.textContent = data.traffic_level || breakdown.traffic_density || 'Medium';

  const distVal = data.osrm_road_distance_km !== undefined && data.osrm_road_distance_km !== null
    ? data.osrm_road_distance_km
    : data.distance_km;
  bDistance.textContent = `${distVal || 0} km`;

  bCity.textContent = breakdown.city_classification || 'Metropolitian';
  bFestival.textContent = (breakdown.festival && breakdown.festival !== 'No') ? `Yes (${breakdown.festival})` : 'No';

  // Fallbacks notification
  if (data.fallbacks_used && data.fallbacks_used.length > 0) {
    const formatted = data.fallbacks_used.map(f => f.replace(/_/g, ' ')).join(', ');
    fallbackText.textContent = `Auto-fallback active: ${formatted}`;
    fallbackBox.classList.remove('hidden');
  } else {
    fallbackBox.classList.add('hidden');
  }

  // Interactive Map
  const pickupCoords = breakdown.pickup_coordinates || data.pickup_coordinates;
  const dropCoords = breakdown.drop_coordinates || data.drop_coordinates;
  if (pickupCoords && dropCoords) {
    initOrUpdateMap(pickupCoords, dropCoords, data.route_geometry);
  }
}

// ==========================================
// 6. UI HELPERS
// ==========================================
function setLoading(isLoading) {
  btnSubmit.disabled = isLoading;
  if (isLoading) {
    btnText.textContent = 'Calculating Route & Deriving Features...';
    spinner.classList.remove('hidden');
  } else {
    btnText.textContent = 'Predict Delivery Time';
    spinner.classList.add('hidden');
  }
}

function showError(title, msg) {
  errorTitle.textContent = title;
  errorMsg.textContent = msg;
  errorAlert.classList.remove('hidden');
}

function hideError() {
  errorAlert.classList.add('hidden');
}

// ==========================================
// 7. PRESET BUTTON HANDLERS
// ==========================================
function setupPresets() {
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      fromInput.value = btn.getAttribute('data-from');
      toInput.value = btn.getAttribute('data-to');
      vehicleTypeSelect.value = btn.getAttribute('data-vehicle') || 'motorcycle';
      typeOfOrderSelect.value = btn.getAttribute('data-order') || 'Meal';
      pickupLatInput.value = '';
      pickupLonInput.value = '';
      dropLatInput.value = '';
      dropLonInput.value = '';

      // Immediately trigger calculation
      handleFormSubmit();
    });
  });
}

// ==========================================
// 8. INITIALIZE ON LOAD
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setupAutocomplete(fromInput, fromSuggestions, pickupLatInput, pickupLonInput);
  setupAutocomplete(toInput, toSuggestions, dropLatInput, dropLonInput);
  setupPresets();
  formEl.addEventListener('submit', handleFormSubmit);
});
