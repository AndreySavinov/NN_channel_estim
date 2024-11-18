
function h = resampleCIR( chan, Ts, n )

    h = zeros( chan.no_rx, chan.no_tx, length(n), chan.no_snap );

    for rxAnt = 1:chan.no_rx
        for snap = 1:chan.no_snap
                
            % path gains is [nTxAnt x nPaths]
            pathGains = squeeze( chan.coeff(rxAnt,:,:,snap) );
            % path delays is [nPaths x 1]
            pathDelays = squeeze( chan.delay(:,snap) );
                
            % repeat n nPath times in vertical dimension
            N = repmat( n, [chan.no_path 1] );
            % repeat delays nTaps times in horizontal dimension
            Delays = repmat( pathDelays, [1 length(n)] );
        
            % for each Tx antenna, each tap is calculated as:
            % h(n) = sum[ pathGain(k) * sinc( pathDelay(k)/Ts - n ) ]
            h(rxAnt,:,:,snap) = pathGains*sinc(Delays/Ts - N);
        end
    end
end
