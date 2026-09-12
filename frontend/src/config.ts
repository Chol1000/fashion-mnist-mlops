/**
 * Deployment-specific links, overridable at build time so a fork doesn't have
 * to edit source. Only used for the "the API is asleep" recovery screen and
 * the About page — nothing functional depends on them.
 */
export const SPACE_URL =
  (import.meta.env.VITE_SPACE_URL as string | undefined) ??
  "https://huggingface.co/spaces/CholatemGiet/fashion-mnist-api";

export const REPO_URL =
  (import.meta.env.VITE_REPO_URL as string | undefined) ??
  "https://github.com/Chol1000/fashion-mnist-mlops";

/** How long a cold start realistically takes on a free CPU Space: container
 *  pull, then TensorFlow import, then loading the MobileNetV2 weights. Used
 *  only to set expectations on the wake-up screen. */
export const COLD_START_SECONDS = 90;
