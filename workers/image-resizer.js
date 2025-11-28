/**
 * Cloudflare Worker for Image Resizing and Optimization
 *
 * This worker uses Cloudflare's built-in Image Resizing feature via cf.image options.
 * Image Resizing is available on paid plans (Pro, Business, Enterprise) or as a $5/month add-on.
 *
 * Enable Image Resizing at: https://dash.cloudflare.com/?to=/:account/images/image-resizing
 *
 * Features:
 * - Dynamic image resizing (?w=800&h=600)
 * - Quality control (?q=85)
 * - Automatic WebP conversion (based on Accept header)
 * - Smart caching (CDN + Browser)
 * - Graceful fallback to original image
 *
 * Example URLs:
 * - Original: https://r2.zhurongshuo.com/images/gallery/photo.jpg
 * - Resized: https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800
 * - Quality: https://r2.zhurongshuo.com/images/gallery/photo.jpg?q=85
 * - Combined: https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&q=75
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Remove leading slash from pathname for R2 object key
    // Decode URI component to handle Chinese characters and other encoded characters
    const pathname = decodeURIComponent(url.pathname.substring(1));

    // Extract query parameters for image processing
    const width = url.searchParams.get('w');
    const height = url.searchParams.get('h');
    const quality = url.searchParams.get('q') || '85';
    const fit = url.searchParams.get('fit') || 'scale-down';

    // Check if client accepts WebP
    const acceptHeader = request.headers.get('Accept') || '';
    const supportsWebP = acceptHeader.includes('image/webp');
    const supportsAVIF = acceptHeader.includes('image/avif');

    // Check if this is an internal subrequest to avoid infinite loops
    const isInternalRequest = request.headers.get('X-Internal-Request') === 'true';

    try {
      // Fetch the original image from R2
      const object = await env.R2_BUCKET.get(pathname);

      if (!object) {
        return new Response('Image not found', { status: 404 });
      }

      // If this is an internal request, just return the original image
      if (isInternalRequest) {
        return new Response(object.body, {
          headers: {
            'Content-Type': object.httpMetadata?.contentType || 'image/jpeg',
            'Cache-Control': 'public, max-age=31536000, immutable',
            'ETag': object.httpEtag,
          }
        });
      }

      // Determine if we should attempt resizing
      const shouldResize = width || height;

      if (shouldResize) {
        try {
          // Build image resizing options
          const imageOptions = {
            quality: parseInt(quality),
            fit: fit,
          };

          // Determine the best format based on browser support
          if (supportsAVIF) {
            imageOptions.format = 'avif';
          } else if (supportsWebP) {
            imageOptions.format = 'webp';
          } else {
            imageOptions.format = 'auto';
          }

          // Add width if specified
          if (width) {
            imageOptions.width = parseInt(width);
          }

          // Add height if specified
          if (height) {
            imageOptions.height = parseInt(height);
          }

          // Build the URL for the internal subrequest (without query params)
          const imageUrl = `https://${url.hostname}${url.pathname}`;

          // Fetch image with Cloudflare Image Resizing
          // Use an internal header to prevent infinite loops
          const resizedResponse = await fetch(imageUrl, {
            headers: {
              'X-Internal-Request': 'true',
              'Accept': request.headers.get('Accept') || '*/*',
            },
            cf: {
              image: imageOptions,
            }
          });

          // Check if the resizing was successful
          if (!resizedResponse.ok) {
            throw new Error(`Image resizing failed with status ${resizedResponse.status}`);
          }

          // Build response headers
          const headers = new Headers();
          headers.set('Content-Type', resizedResponse.headers.get('Content-Type') || 'image/jpeg');
          headers.set('Cache-Control', 'public, max-age=31536000, immutable');
          headers.set('Access-Control-Allow-Origin', '*');
          headers.set('Vary', 'Accept');
          headers.set('X-Image-Processing', 'resized');
          headers.set('X-Image-Width-Requested', width || 'auto');
          headers.set('X-Image-Height-Requested', height || 'auto');
          headers.set('X-Image-Quality-Requested', quality);
          headers.set('X-Image-Fit', fit);
          headers.set('X-Image-Format', imageOptions.format);

          // Copy CF-specific headers if they exist
          if (resizedResponse.headers.get('CF-Cache-Status')) {
            headers.set('CF-Cache-Status', resizedResponse.headers.get('CF-Cache-Status'));
          }
          if (resizedResponse.headers.get('CF-Image-Format')) {
            headers.set('CF-Image-Format', resizedResponse.headers.get('CF-Image-Format'));
          }

          return new Response(resizedResponse.body, {
            status: resizedResponse.status,
            headers: headers
          });

        } catch (resizeError) {
          console.error('Image resizing error, falling back to original:', resizeError);

          // Fall back to original image if resizing fails
          return new Response(object.body, {
            headers: {
              'Content-Type': object.httpMetadata?.contentType || 'image/jpeg',
              'Cache-Control': 'public, max-age=31536000, immutable',
              'Access-Control-Allow-Origin': '*',
              'Vary': 'Accept',
              'X-Image-Processing': 'fallback-to-original',
              'X-Image-Error': resizeError.message,
            }
          });
        }
      }

      // Return original image without processing
      return new Response(object.body, {
        headers: {
          'Content-Type': object.httpMetadata?.contentType || 'image/jpeg',
          'Cache-Control': 'public, max-age=31536000, immutable',
          'Access-Control-Allow-Origin': '*',
          'ETag': object.httpEtag,
          'X-Image-Processing': 'original',
        }
      });

    } catch (error) {
      console.error('Error processing image:', error);
      return new Response('Error processing image: ' + error.message, {
        status: 500
      });
    }
  }
};
