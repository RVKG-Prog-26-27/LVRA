import { ApiRecognitionService } from './apiRecognitionService';

const image = new Blob([new Uint8Array([1, 2, 3])], { type: 'image/jpeg' });
const signal = () => new AbortController().signal;

const jsonResponse = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

describe('ApiRecognitionService', () => {
  it('posts the image to <base>/recognize and returns the text', async () => {
    const fetchMock = vi.fn(async () => jsonResponse(200, { text: 'Sveiki', words: [] }));
    const service = new ApiRecognitionService('https://server.example/api/', fetchMock);

    const result = await service.recognize(image, { signal: signal() });

    expect(result.text).toBe('Sveiki');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('https://server.example/api/recognize');
    expect(init.method).toBe('POST');
    expect((init.body as FormData).get('image')).toBeInstanceOf(Blob);
  });

  it('maps server errors to user-friendly codes', async () => {
    const service = new ApiRecognitionService('https://s', async () =>
      jsonResponse(422, { error: { code: 'NO_TEXT_FOUND' } }),
    );
    await expect(service.recognize(image, { signal: signal() })).rejects.toMatchObject({ code: 'NO_TEXT_FOUND' });
  });

  it('reports network failures as service unavailable', async () => {
    const service = new ApiRecognitionService('https://s', async () => {
      throw new TypeError('Failed to fetch');
    });
    await expect(service.recognize(image, { signal: signal() })).rejects.toMatchObject({
      code: 'SERVICE_UNAVAILABLE',
    });
  });

  it('reports a missing server address as service unavailable', async () => {
    const service = new ApiRecognitionService('');
    await expect(service.recognize(image, { signal: signal() })).rejects.toMatchObject({
      code: 'SERVICE_UNAVAILABLE',
    });
  });
});
