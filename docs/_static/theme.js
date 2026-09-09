/* Keep Furo's controls, but offer only paper (light) and ink (dark). */
(() => {
    const buttons = document.querySelectorAll(".theme-toggle");

    function setTheme(theme) {
        document.body.dataset.theme = theme;
        try {
            localStorage.setItem("theme", theme);
        } catch {
            // Theme switching still works when browser storage is unavailable.
        }
        const label = `Switch to ${theme === "dark" ? "light" : "dark"} mode`;
        buttons.forEach((button) => {
            button.setAttribute("aria-label", label);
            button.setAttribute("title", label);
        });
    }

    // Resolve an old "auto" preference once; it is never part of the cycle.
    const initial = document.body.dataset.theme;
    setTheme(initial === "dark" || (initial === "auto" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light");

    buttons.forEach((button) => {
        button.addEventListener("click", (event) => {
            // Prevent Furo's three-state click handler from running as well.
            event.stopImmediatePropagation();
            setTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
        }, {capture: true});
    });
})();
