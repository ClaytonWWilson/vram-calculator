import { json } from "@sveltejs/kit";

import { resolveModelPayload } from "$lib/server/huggingface";

export async function POST({ request, fetch }) {
  let body;

  try {
    body = await request.json();
  } catch {
    return json({ error: "Send JSON with a repo field." }, { status: 400 });
  }

  try {
    const payload = await resolveModelPayload(body.repo, fetch);
    return json(payload);
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Unable to resolve model metadata.";
    return json({ error: message }, { status: 400 });
  }
}
