import { mount } from "@cloudflare/nimbus-docs/client";

function initPageActions(root: HTMLElement): () => void {
  const copyBtn = root.querySelector<HTMLButtonElement>("[data-nb-page-actions-copy]");
  const copyIcon = root.querySelector<SVGElement>("[data-nb-page-actions-copy-icon]");
  const checkIcon = root.querySelector<SVGElement>("[data-nb-page-actions-check-icon]");
  const label = root.querySelector<HTMLSpanElement>("[data-nb-page-actions-label]");
  const mdUrl = root.dataset.mdUrl;

  if (!copyBtn || !mdUrl) return () => {};

  let resetTimer: number | undefined;

  function showState(state: "copied" | "error") {
    if (!copyIcon || !checkIcon || !label) return;
    if (state === "copied") {
      copyIcon.classList.add("hidden");
      checkIcon.classList.remove("hidden");
      label.textContent = "Copied";
    } else {
      label.textContent = "Couldn't copy";
    }
    if (resetTimer) window.clearTimeout(resetTimer);
    resetTimer = window.setTimeout(() => {
      copyIcon.classList.remove("hidden");
      checkIcon.classList.add("hidden");
      label.textContent = "Copy page";
    }, 1500);
  }

  async function handleCopyPage() {
    try {
      if (typeof ClipboardItem !== "undefined") {
        // Call clipboard.write() synchronously in the gesture handler.
        // Pass a Promise<Blob> as the item's value — Safari will wait for it.
        const item = new ClipboardItem({
          "text/plain": fetch(mdUrl!).then(async (res) => {
            if (!res.ok) throw new Error("fetch failed");
            const text = await res.text();
            return new Blob([text], { type: "text/plain" });
          }),
        });
        await navigator.clipboard.write([item]);
      } else {
        // Fallback: no way around this being async, this path just won't work
        // reliably on Safari if it doesn't support ClipboardItem promises.
        const res = await fetch(mdUrl!);
        const text = await res.text();
        await navigator.clipboard.writeText(text);
      }
      showState("copied");
    } catch {
      showState("error");
    }
  }

  copyBtn.addEventListener("click", handleCopyPage);

  return () => {
    if (resetTimer) window.clearTimeout(resetTimer);
    copyBtn.removeEventListener("click", handleCopyPage);
  };
}

mount("[data-nb-page-actions]", initPageActions);
