/* Shared query filters. All listing selection and numeric validation live in Python. */
(() => {
  "use strict";

  const names = ["property_type", "l2", "l4", "min_price", "max_price", "min_area", "max_area", "bedrooms", "bathrooms", "q"];
  const optionRequests = new WeakMap();

  function params(form) {
    const query = new URLSearchParams();
    names.forEach((name) => {
      const control = form.elements.namedItem(name);
      const value = control ? control.value.trim() : "";
      if (value !== "") query.set(name, value);
    });
    return query;
  }

  function fillOptions(select, values, label, preserve) {
    const selected = preserve === undefined ? select.value : preserve;
    const fragment = document.createDocumentFragment();
    const all = document.createElement("option");
    all.value = "";
    all.textContent = label;
    fragment.append(all);
    const choices = [...new Set(values.filter((value) => typeof value === "string" && value.trim()))];
    if (selected && !choices.includes(selected)) choices.push(selected);
    choices.forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value === "Casa" ? "Casa — House" : value === "Departamento" ? "Departamento — Apartment" : value;
      fragment.append(option);
    });
    select.replaceChildren(fragment);
    select.value = selected;
  }

  async function loadOptions(form) {
    const previous = optionRequests.get(form);
    if (previous) previous.abort();
    const controller = new AbortController();
    optionRequests.set(form, controller);
    const note = form.querySelector("#filter-options-status");
    if (note) note.textContent = "Loading location options…";
    const query = new URLSearchParams();
    const region = form.elements.namedItem("l2").value;
    if (region) query.set("l2", region);
    try {
      const response = await fetch(`/api/options${query.size ? `?${query}` : ""}`, {signal: controller.signal, headers: {Accept: "application/json"}});
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.error === "string" ? data.error : "Location options are unavailable.");
      if (controller.signal.aborted) return null;
      fillOptions(form.elements.namedItem("property_type"), data.property_types || [], "All property types");
      fillOptions(form.elements.namedItem("l2"), data.regions || [], "All regions");
      fillOptions(form.elements.namedItem("l4"), data.localities || [], "All localities");
      if (note) note.textContent = "";
      return data;
    } catch (error) {
      if (error.name !== "AbortError" && note) note.textContent = error.message || "Location options are unavailable. Try again after preparing the local data.";
      return null;
    } finally {
      if (optionRequests.get(form) === controller) optionRequests.delete(form);
    }
  }

  function initializeQuery(form) {
    const initial = new URLSearchParams(window.location.search);
    names.forEach((name) => {
      const value = initial.get(name);
      const control = form.elements.namedItem(name);
      if (value === null || !control) return;
      if (control.tagName === "SELECT" && ![...control.options].some((option) => option.value === value)) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        control.append(option);
      }
      control.value = value;
    });
  }

  function validateRanges(form) {
    let valid = true;
    [["min_price", "max_price"], ["min_area", "max_area"]].forEach(([low, high]) => {
      const minimum = form.elements.namedItem(low);
      const maximum = form.elements.namedItem(high);
      maximum.setCustomValidity("");
      if (minimum.value !== "" && maximum.value !== "" && Number(minimum.value) > Number(maximum.value)) {
        maximum.setCustomValidity("The maximum must be greater than or equal to the minimum.");
        valid = false;
      }
    });
    return valid;
  }

  function bind(form, onApply) {
    initializeQuery(form);
    const apply = () => {
      const query = params(form);
      const url = new URL(window.location.href);
      url.search = query.toString();
      window.history.replaceState({}, "", url);
      onApply(query);
    };
    form.addEventListener("input", () => validateRanges(form));
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (!validateRanges(form) || !form.reportValidity()) return;
      apply();
    });
    form.addEventListener("reset", () => {
      // Apply after the browser has restored the controls' default values.
      window.setTimeout(() => {
        names.forEach((name) => { form.elements.namedItem(name).value = ""; });
        validateRanges(form);
        loadOptions(form);
        apply();
      }, 0);
    });
    form.elements.namedItem("l2").addEventListener("change", () => {
      form.elements.namedItem("l4").value = "";
      loadOptions(form);
    });
  }

  window.ExplorerFilters = {params, loadOptions, bind};
})();
