// Thin client for the DooDL HTTP API.

const hashParams = new URLSearchParams(location.hash.slice(1));
export const token = hashParams.get('token') || '';

export class ApiError extends Error {
  constructor(message, code = 'error', status = 0) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

/** Build a same-origin URL, adding the access token when this link has one. */
export function url(path, params = {}) {
  const u = new URL(path, location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) u.searchParams.set(key, value);
  }
  if (token) u.searchParams.set('token', token);
  return u.pathname + u.search;
}

async function parse(response) {
  if (response.ok) return response.status === 204 ? null : response.json();
  let message = `The server answered ${response.status}.`;
  let code = 'error';
  try {
    const body = await response.json();
    if (body && body.error) {
      message = body.error.message;
      code = body.error.code;
    }
  } catch {
    /* not JSON: keep the generic message */
  }
  throw new ApiError(message, code, response.status);
}

async function request(method, path, body) {
  let response;
  try {
    response = await fetch(url(path), {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError('Could not reach the server. Check that it is still running.', 'network');
  }
  return parse(response);
}

export const get = (path) => request('GET', path);
export const post = (path, body) => request('POST', path, body ?? {});
export const del = (path) => request('DELETE', path);

/** Upload a video as the raw request body so the server can enforce the size limit while streaming. */
export function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url('/api/sessions', { filename: file.name }));
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    xhr.onload = () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* fall through */
      }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      const err = body && body.error;
      reject(new ApiError(err ? err.message : `The server answered ${xhr.status}.`, err ? err.code : 'error', xhr.status));
    };
    xhr.onerror = () => reject(new ApiError('The upload was interrupted. Check your connection and try again.', 'network'));
    xhr.send(file);
  });
}
